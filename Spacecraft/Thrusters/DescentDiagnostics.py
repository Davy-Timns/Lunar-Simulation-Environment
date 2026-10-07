"""Save interpretable end-of-run metrics independently of optimizer acceptance."""
import json
from pathlib import Path
import numpy as np
from Basilisk.architecture import sysModel
from Basilisk.utilities import RigidBodyKinematics


def touchdown_metrics(position, velocity, target_position, terrain, fuel_remaining,
                      initial_fuel, time_s, ignition_s, braking_s, terminal_s):
    """First-contact geometry/velocity, with local vertical positive upward."""
    position, velocity = np.asarray(position), np.asarray(velocity)
    map_position = terrain.to_map(position)
    map_target = terrain.to_map(target_position)
    # Coordinate axes are fixed by this terrain, not by vehicle body attitude.
    velocity_map = terrain.to_map(position + velocity) - map_position
    speed = float(np.linalg.norm(velocity_map))
    direction = velocity_map / speed if speed > 1e-12 else np.zeros(3)
    site = terrain.assess_site(map_position[:2], radius_m=3.)
    return dict(time_s=float(time_s), intended_site_map_xy_m=map_target[:2].tolist(),
        touchdown_site_map_xy_m=map_position[:2].tolist(),
        distance_to_intended_site_m=float(np.linalg.norm(map_position[:2] - map_target[:2])),
        fuel_remaining_kg=float(fuel_remaining), fuel_used_kg=float(initial_fuel - fuel_remaining),
        landed_area_slope_deg=site.get('slope_deg'),
        ignition_to_touchdown_s=float(time_s - ignition_s),
        braking_to_touchdown_s=float(time_s - braking_s),
        terminal_to_touchdown_s=None if terminal_s is None else float(time_s - terminal_s),
        velocity_N_m_s=velocity.tolist(), velocity_map_m_s=velocity_map.tolist(),
        speed_m_s=speed, direction_map_unit=direction.tolist(),
        horizontal_velocity_map_m_s=velocity_map[:2].tolist(),
        horizontal_speed_m_s=float(np.linalg.norm(velocity_map[:2])),
        vertical_velocity_m_s=float(velocity_map[2]),
        descent_angle_deg=float(np.rad2deg(np.arctan2(-velocity_map[2], np.linalg.norm(velocity_map[:2])))),
        horizontal_heading_deg=(float(np.rad2deg(np.arctan2(velocity_map[1], velocity_map[0])))
                                if np.linalg.norm(velocity_map[:2]) > 1e-12 else None))


def print_touchdown_metrics(event):
    def number(value):
        return 'unavailable' if value is None else f'{value:.3f}'
    print('\nTouchdown diagnostics (t0 = first detected foot contact)', flush=True)
    print(f"  Distance to intended landing site: {event['distance_to_intended_site_m']:.3f} m", flush=True)
    print(f"  Fuel remaining: {event['fuel_remaining_kg']:.3f} kg; used: {event['fuel_used_kg']:.3f} kg", flush=True)
    print(f"  Maximum acceleration including contact so far: {event['maximum_acceleration_m_s2']:.3f} m/s^2", flush=True)
    print(f"  Maximum engine acceleration: {event['maximum_engine_acceleration_m_s2']:.3f} m/s^2 "
          f"(main alone {event['maximum_main_engine_acceleration_m_s2']:.3f} m/s^2)", flush=True)
    print(f"  Landed area slope: {number(event['landed_area_slope_deg'])} deg", flush=True)
    print(f"  Ignition / braking / terminal to t0: {event['ignition_to_touchdown_s']:.3f} / "
          f"{event['braking_to_touchdown_s']:.3f} / {number(event['terminal_to_touchdown_s'])} s", flush=True)
    print(f"  Touchdown velocity: {event['speed_m_s']:.3f} m/s; local XYZ {event['velocity_map_m_s']} m/s", flush=True)
    print(f"  Horizontal: {event['horizontal_speed_m_s']:.3f} m/s, XY {event['horizontal_velocity_map_m_s']} m/s; "
          f"vertical (+up): {event['vertical_velocity_m_s']:.3f} m/s", flush=True)
    print(f"  Direction (local unit vector): {event['direction_map_unit']}; "
          f"descent angle {event['descent_angle_deg']:.3f} deg; "
          f"horizontal heading {number(event['horizontal_heading_deg'])} deg", flush=True)


class TouchdownDiagnostics(sysModel.SysModel):
    """Observe actual first contact and loads; never supplies controller feedback."""
    def __init__(self, vehicle, print_on_contact=True):
        super().__init__()
        self.ModelTag = 'TouchdownDiagnostics'
        self.vehicle = vehicle
        self.print_on_contact = print_on_contact
        self.state = vehicle.lander.scStateOutMsg.addSubscriber()
        self.mass = vehicle.lander.scMassOutMsg.addSubscriber()
        contact = vehicle.thruster_controller.contact_sensor
        self.contact_force = contact.forceOutMsg.addSubscriber() if contact is not None else None
        imu = vehicle.imus.get('IMU')
        self.imu = imu.model.sensorOutMsg.addSubscriber() if imu is not None else None
        self.accel_bias = np.asarray(imu.model.senTransBias).reshape(3) if imu is not None else np.zeros(3)
        self.Reset(0)

    def Reset(self, now):
        self.touchdown = None
        self.history = []
        self.peaks = dict(maximum_acceleration_m_s2=0., maximum_imu_acceleration_m_s2=0.,
                          maximum_engine_acceleration_m_s2=0., maximum_main_engine_acceleration_m_s2=0.,
                          maximum_contact_acceleration_m_s2=0.)

    def UpdateState(self, now):
        vehicle = self.vehicle
        controller = vehicle.thruster_controller
        t = now * 1e-9
        ignition = controller.start_s + controller.settings.coast_duration_s
        if t < ignition or not self.state.isWritten():
            return
        state = self.state()
        mass = float(self.mass().massSC) if self.mass.isWritten() else controller.initial_state.mass
        c_NB = RigidBodyKinematics.MRP2C(state.sigma_BN).T
        engine_force = sum((np.asarray(e.thruster.forceExternal_B).reshape(3)
                            for e in vehicle.engines.values()), np.zeros(3))
        main_force = np.asarray(controller.main_engine.thruster.forceExternal_B).reshape(3)
        contact = controller.contact_sensor
        contact_force = (np.asarray(self.contact_force().forceRequestInertial)
                         if self.contact_force is not None and self.contact_force.isWritten() else np.zeros(3))
        specific = float(np.linalg.norm(c_NB @ engine_force + contact_force) / mass)
        imu_accel = 0.
        if self.imu is not None and self.imu.isWritten():
            imu_accel = float(np.linalg.norm(np.asarray(self.imu().AccelPlatform) - self.accel_bias))
        values = dict(maximum_acceleration_m_s2=specific,
                      maximum_imu_acceleration_m_s2=imu_accel,
                      maximum_engine_acceleration_m_s2=float(np.linalg.norm(engine_force) / mass),
                      maximum_main_engine_acceleration_m_s2=float(np.linalg.norm(main_force) / mass),
                      maximum_contact_acceleration_m_s2=float(np.linalg.norm(contact_force) / mass))
        for key, value in values.items():
            self.peaks[key] = max(self.peaks[key], value)
        self.history.append(dict(time_s=t, specific_acceleration_m_s2=specific,
            imu_acceleration_m_s2=imu_accel, engine_acceleration_m_s2=values['maximum_engine_acceleration_m_s2'],
            contact_acceleration_m_s2=values['maximum_contact_acceleration_m_s2']))
        if self.touchdown is None and contact is not None and contact.collision:
            fuel = float(vehicle.fuel_tank.fuelTankOutMsg.read().fuelMass)
            target = controller.target.state(t)[0]
            self.touchdown = touchdown_metrics(state.r_BN_N, state.v_BN_N, target,
                vehicle.terrain_map, fuel, controller.initial_state.mass - controller.settings.dry_mass,
                t, ignition, ignition, controller.terminal_start_s)
            self.touchdown.update(self.peaks)
            self.touchdown['cutoff'] = controller.cutoff_event
            self.touchdown['acceleration_definition'] = 'specific load |engine + contact force| / mass; gravity excluded'
            if self.print_on_contact:
                print_touchdown_metrics(self.touchdown)

    def report(self):
        if self.touchdown is None:
            return None
        event = dict(self.touchdown)
        event['peaks_at_first_contact'] = {key: self.touchdown[key] for key in self.peaks}
        # Contact-load peak may occur after the first contact detection. Keep
        # first-contact velocity/fuel/time fixed while updating load maxima.
        event.update(self.peaks)
        event['acceleration_peak_interval'] = 'powered ignition through end of settling'
        return event


def terminal_metrics(position, velocity, target_position, target_velocity, radius):
    r = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)
    er = float(np.linalg.norm(r - target_position))
    ev = float(np.linalg.norm(v - target_velocity))
    return dict(position_error_m=er, velocity_error_m_s=ev,
                reference_altitude_m=float(np.linalg.norm(r)-radius),
                speed_m_s=float(np.linalg.norm(v)),
                radial_speed_m_s=float(np.dot(r,v)/np.linalg.norm(r)),
                precision_target_met=bool(er <= .5 and ev <= .1),
                contact_landing_verified=False)


def save_descent_report(vehicle, directory):
    import matplotlib.pyplot as plt
    from Basilisk.utilities import RigidBodyKinematics
    controller = vehicle.thruster_controller
    cfg = controller.settings
    time = np.asarray(vehicle.sc_recorder.times())*1e-9
    r = np.asarray(vehicle.sc_recorder.r_BN_N)
    v = np.asarray(vehicle.sc_recorder.v_BN_N)
    target = controller.target.state(time[-1])
    report = terminal_metrics(r[-1],v[-1],*target[:2],cfg.surface_radius)
    altitude = np.linalg.norm(r,axis=1)-cfg.surface_radius
    report.update(controller_status=controller.status,
                  simulation_start_s=getattr(vehicle, 'simulation_start_s', 0.),
                  initialization=('powered-descent ignition; sensors initialized at ignition'
                      if getattr(vehicle, 'simulation_start_s', 0.) else 'full mission'),
                  configured_position_tolerance_m=cfg.position_tolerance,
                  configured_velocity_tolerance_m_s=cfg.velocity_tolerance,
                  minimum_reference_altitude_m=float(np.min(altitude)),
                  powered_flight_time_s=controller.plan.flight_time,
                  dry_mass_kg=cfg.dry_mass,
                  engine_cutoff_m=getattr(cfg, 'engine_cutoff_m', 0.),
                  main_engine_cutoff_triggered=getattr(controller, 'main_engine_cutoff', False),
                  remaining_propellant_kg=float(vehicle.fuel_recorder.fuelMass[-1]))
    report['precision_target_met'] = bool(report['position_error_m'] <= cfg.position_tolerance
                                          and report['velocity_error_m_s'] <= cfg.velocity_tolerance)
    contact = getattr(controller, 'contact_sensor', None)
    report['contact_detected_at_end'] = bool(contact is not None and contact.collision)
    report['contact_landing_verified'] = bool(controller.status == 'landed'
        and report['contact_detected_at_end'] and report['precision_target_met'])
    monitor = getattr(vehicle, 'touchdown_diagnostics', None)
    report['touchdown'] = monitor.report() if monitor is not None else None
    report['cutoff'] = getattr(controller, 'cutoff_event', None)
    c_NB=RigidBodyKinematics.MRP2C(vehicle.sc_recorder.sigma_BN[-1]).T
    report['final_body_tilt_deg']=float(np.rad2deg(np.arccos(np.clip(
        np.dot(c_NB[:,2],r[-1]/np.linalg.norm(r[-1])),-1,1))))
    report['final_angular_speed_rad_s']=float(np.linalg.norm(vehicle.sc_recorder.omega_BN_B[-1]))
    if hasattr(vehicle, 'terrain_map'):
        report['final_terrain_clearance_m'] = vehicle.terrain_map.clearance(r[-1])
        report['landing_site'] = getattr(vehicle, 'landing_site', None)
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    navigation = getattr(vehicle,'navigation',None)
    if navigation is not None and navigation.history:
        nh = navigation.history
        nt = np.array([h['time_s'] for h in nh])
        nr = np.array([h['position_N'] for h in nh])
        nv = np.array([h['velocity_N'] for h in nh])
        nr_true = np.column_stack([np.interp(nt,time,r[:,i]) for i in range(3)])
        nv_true = np.column_stack([np.interp(nt,time,v[:,i]) for i in range(3)])
        errors = np.linalg.norm(nr-nr_true,axis=1)
        valid = np.array([h['range_valid'] for h in nh])
        report['navigation'] = dict(all_healthy=all(h['healthy'] for h in nh),
            filter_enabled=navigation.cfg.filter_enabled,
            final_position_error_m=float(errors[-1]), maximum_position_error_m=float(errors.max()),
            laser_updates=int(valid.sum()),
            first_laser_update_s=float(nt[valid][0]) if valid.any() else None,
            inputs='Gauss-Markov map XY, map-referenced laser range, star tracker, IMU, commanded thrust model')
        if not navigation.cfg.filter_enabled:
            report['navigation']['inputs']='Raw Gauss-Markov map XY and map-referenced laser fixes; calibrated IMU dead reckoning during outages'
        status=np.array([h['range_status'] for h in nh])
        reasons,counts=np.unique(status,return_counts=True)
        report['navigation']['laser_status_counts']=dict(zip(reasons.tolist(),counts.tolist()))
        laser_fields=('raw_slant_range_m','used_slant_range_m','laser_observed_map_z_m',
                      'predicted_map_z_m','estimated_map_z_m','beam_down_cosine','measured_map_xy_m')
        laser_data={key:np.array([h[key] for h in nh]) for key in laser_fields}
        np.savez_compressed(directory/'navigation.npz',time_s=nt,position_N=nr,velocity_N=nv,
            truth_position_N=nr_true,truth_velocity_N=nv_true,range_valid=valid,
            position_sigma_m=[h['position_sigma_m'] for h in nh],range_status=status,
            filter_enabled=navigation.cfg.filter_enabled,**laser_data)
        laserfig,laseraxes=plt.subplots(3,1,figsize=(11,9),layout='constrained',sharex=True)
        lt=(nt-cfg.coast_duration_s)/60
        raw=laser_data['raw_slant_range_m'].copy()
        raw[(raw<0)|~np.isfinite(raw)]=np.nan
        laseraxes[0].plot(lt,raw,label='Raw return',alpha=.65)
        laseraxes[0].plot(lt,laser_data['used_slant_range_m'],label='Accepted noisy range')
        laseraxes[0].set(ylabel='Beam slant range (m)')
        laseraxes[1].plot(lt,laser_data['estimated_map_z_m'],label='Navigation map Z')
        laseraxes[1].plot(lt,laser_data['laser_observed_map_z_m'],'.',ms=2,label='Laser + DEM observed Z')
        if hasattr(vehicle,'terrain_map'):
            laseraxes[1].plot(lt,vehicle.terrain_map.to_map(nr_true)[:,2],':',label='Truth Z (comparison only)')
        laseraxes[1].set(ylabel='Map Z (m)')
        codes={name:i for i,name in enumerate(('no_return','stale','over_range','beam_not_downward',
                                              'footprint_unmapped','negative_noisy_range','accepted'))}
        laseraxes[2].step(lt,[codes[s] for s in status],where='post',label='Navigation range decision')
        laseraxes[2].set(yticks=list(codes.values()),yticklabels=list(codes),xlabel='Time since ignition (min)')
        for ax in laseraxes:
            ax.grid(alpha=.25);ax.legend()
        laserfig.suptitle('Laser measurement use — navigation predicts Z whenever a range is rejected')
        laserfig.savefig(directory/'laser_navigation_diagnostics.png',dpi=150)
        # A second view prevents the long coast from hiding the useful returns.
        if valid.any():
            laseraxes[2].set_xlim(lt[valid][0]-.1,lt[-1]+.1)
            z_values=np.r_[laser_data['estimated_map_z_m'][valid],laser_data['laser_observed_map_z_m'][valid]]
            margin=max(1.,float(np.ptp(z_values))*.05)
            laseraxes[1].set_ylim(float(np.min(z_values))-margin,float(np.max(z_values))+margin)
            laseraxes[0].set_ylim(0,max(1.,float(np.max(laser_data['used_slant_range_m'][valid]))*1.1))
            laserfig.savefig(directory/'laser_navigation_closeup.png',dpi=150)
        plt.close(laserfig)
        navfig,navaxes=plt.subplots(2,1,figsize=(9,6),layout='constrained')
        navaxes[0].plot((nt-cfg.coast_duration_s)/60,errors)
        navaxes[0].set(ylabel='Position estimate error (m)',yscale='symlog')
        navaxes[1].plot((nt-cfg.coast_duration_s)/60,valid.astype(int))
        navaxes[1].set(ylabel='Laser update valid',yticks=[0,1],xlabel='Time since ignition (min)')
        for ax in navaxes: ax.grid(alpha=.25)
        navfig.savefig(directory/'navigation_diagnostics.png',dpi=150)
        plt.close(navfig)
    attitude = getattr(vehicle, 'attitude_navigation', None)
    if attitude is not None and attitude.history:
        ah = attitude.history
        report['attitude_navigation'] = dict(inputs='calibrated IMU gyros + star-tracker drift correction',
            all_healthy=all(h['healthy'] for h in ah), samples=len(ah))
        np.savez_compressed(directory/'attitude_navigation.npz',
            time_s=[h['time_s'] for h in ah], dcm_NB=[h['dcm_NB'] for h in ah],
            omega_B_rad_s=[h['omega_B_rad_s'] for h in ah])
    if report['touchdown'] is not None:
        print_touchdown_metrics(report['touchdown'])
    if monitor is not None and monitor.history:
        np.savez_compressed(directory/'touchdown_acceleration.npz',
            **{key: np.array([h[key] for h in monitor.history]) for key in monitor.history[0]})
    (directory/'descent_report.json').write_text(json.dumps(report,indent=2))
    fig,axes = plt.subplots(3,2,figsize=(11,10),layout='constrained')
    powered = time >= cfg.coast_duration_s
    t = (time[powered]-cfg.coast_duration_s)/60
    axes[0,0].plot(t,altitude[powered]); axes[0,0].axhline(0,color='black',lw=.8)
    axes[0,0].set(ylabel='Reference altitude (m)')
    up = r / np.linalg.norm(r,axis=1)[:,None]
    radial = np.sum(v*up,axis=1)
    horizontal = np.linalg.norm(v-radial[:,None]*up,axis=1)
    axes[0,1].plot(t,radial[powered],label='Radial (+ upward)')
    axes[0,1].plot(t,horizontal[powered],label='Horizontal')
    axes[0,1].set(ylabel='Velocity (m/s)'); axes[0,1].legend()
    history = [h for h in controller.history if 'main_thrust' in h]
    axes[1,0].plot([(h['time_s']-cfg.coast_duration_s)/60 for h in history],
                   [h['main_thrust'] for h in history])
    axes[1,0].set(ylabel='Main thrust (N)')
    ht = np.array([(h['time_s']-cfg.coast_duration_s)/60 for h in history])
    for field,label in [('requested_radial_thrust_N','Requested'),('applied_radial_thrust_N','Applied')]:
        axes[1,1].plot(ht,[h.get(field, np.nan) for h in history],label=label)
    axes[1,1].axhline(0,color='black',lw=.8)
    axes[1,1].set(ylabel='Radial thrust (+ upward, N)',yscale='symlog')
    axes[1,1].legend()
    ft = np.asarray(vehicle.fuel_recorder.times())*1e-9
    fm = np.asarray(vehicle.fuel_recorder.fuelMass)
    axes[2,0].plot((ft[ft>=cfg.coast_duration_s]-cfg.coast_duration_s)/60,fm[ft>=cfg.coast_duration_s])
    axes[2,0].set(ylabel='Propellant remaining (kg)')
    axes[2,1].plot(ht,[h.get('position_error', np.nan) for h in history])
    axes[2,1].set(ylabel='Target position error (m)')
    keys = ('time_s','reference_altitude_m','radial_speed_m_s','horizontal_speed_m_s',
            'requested_radial_thrust_N','applied_radial_thrust_N','force_error_N','main_thrust','mass')
    np.savez_compressed(directory/'descent_guidance.npz',
                        **{key: np.array([h.get(key, np.nan) for h in history]) for key in keys})
    for ax in axes.flat:
        ax.set_xlabel('Time since ignition (min)'); ax.grid(alpha=.25)
    verdict = ('Contact landing verified in simulation' if report['contact_landing_verified'] else
               'Precision target met; contact unverified' if report['precision_target_met'] else 'Landing target NOT achieved')
    fig.suptitle(f"{verdict}: error {report['position_error_m']:.2f} m, {report['velocity_error_m_s']:.2f} m/s")
    fig.savefig(directory/'descent_diagnostics.png',dpi=160)
    plt.close(fig)
    print(f"Descent assessment: {verdict}. Position error {report['position_error_m']:.3f} m; "
          f"velocity error {report['velocity_error_m_s']:.3f} m/s.",flush=True)
    print('Descent report:',(directory/'descent_report.json').resolve(),flush=True)
    return report
