"""Noisy map X/Y + laser-range navigation using Basilisk's SmallBodyNavUKF.

Truth is used only by the horizontal measurement simulator and initial prior.
Laser altitude is reconstructed from a known map and measured star-tracker
attitude. Accelerometers update the UKF's non-gravitational acceleration state.
The live run uses IMU-propagated, star-tracker-corrected attitude for both
navigation reconstruction and controller attitude/rate feedback.
"""
from dataclasses import dataclass
import numpy as np
from Basilisk.architecture import messaging, sysModel
from Basilisk.fswAlgorithms import smallBodyNavUKF
from Basilisk.utilities import RigidBodyKinematics
from Environment.TerrainMap import MAP_TO_N


class GaussMarkovXY:
    def __init__(self, sigma_m=.25, correlation_s=10., seed=42):
        if sigma_m < 0 or correlation_s <= 0:
            raise ValueError('Noise sigma must be nonnegative and correlation time positive')
        self.sigma=sigma_m
        self.tau=correlation_s
        self.rng=np.random.default_rng(seed)
        self.error=self.rng.normal(0,sigma_m,2)

    def sample(self,dt):
        phi=np.exp(-max(0.,dt)/self.tau)
        self.error=phi*self.error+self.sigma*np.sqrt(1-phi*phi)*self.rng.normal(size=2)
        return self.error.copy()


@dataclass
class NavigationSettings:
    filter_enabled: bool = True
    period_s: float = .1
    xy_sigma_m: float = .25
    xy_correlation_s: float = 10.
    range_sigma_m: float = .05
    max_range_m: float = 5000.
    minimum_down_cosine: float = .5
    attitude_sigma_rad: float = .004
    accel_sigma_m_s2: float = .09  # existing IMU PMatrixAccel has diagonal .0885
    seed: int = 42

    def __post_init__(self):
        if not isinstance(self.filter_enabled, bool):
            raise ValueError('filter_enabled must be a boolean fixed at simulation startup')
        if not np.all(np.isfinite(list(vars(self).values()))):
            raise ValueError('Navigation settings must be finite')
        if min(self.period_s,self.xy_correlation_s,self.max_range_m,self.accel_sigma_m_s2) <= 0:
            raise ValueError('Navigation periods, range and acceleration sigma must be positive')
        if min(self.xy_sigma_m,self.range_sigma_m,self.attitude_sigma_rad) < 0 or not 0 < self.minimum_down_cosine <= 1:
            raise ValueError('Invalid sensor noise or laser down-cone settings')


def predict_sigma_points(x, p, dt, mu):
    """RK4 propagation with the same alpha=1, beta=2, kappa=0 UKF weights.

    The native module's Euler step creates orbital drift during laser outages.
    Propagate the distribution here, then use the native UKF's measurement
    update at zero elapsed filter time. Public NavTrans time remains real time.
    """
    if dt <= 0:
        return x,p
    root=np.linalg.cholesky(.5*(p+p.T)+np.eye(9)*1e-12)*3.
    sigma=np.vstack((x,x+root.T,x-root.T))
    def derivative(states):
        result=np.zeros_like(states)
        result[:,:3]=states[:,3:6]
        radius=np.linalg.norm(states[:,:3],axis=1)
        result[:,3:6]=-mu*states[:,:3]/radius[:,None]**3+states[:,6:]
        return result
    k1=derivative(sigma)
    k2=derivative(sigma+.5*dt*k1)
    k3=derivative(sigma+.5*dt*k2)
    k4=derivative(sigma+dt*k3)
    propagated=sigma+dt/6*(k1+2*k2+2*k3+k4)
    mean=np.mean(propagated[1:],axis=0)
    delta=propagated-mean
    covariance=delta[1:].T@delta[1:]/18+2*np.outer(delta[0],delta[0])
    return mean,.5*(covariance+covariance.T)


def position_from_range(terrain, xy, range_m, c_NB, offset_B=(0.,0.,-.3), direction_B=(0.,0.,-1.)):
    """Convert SLANT range to COM map height, including beam footprint/offset."""
    offset_M=MAP_TO_N.T @ c_NB @ np.asarray(offset_B)
    direction_M=MAP_TO_N.T @ c_NB @ np.asarray(direction_B)
    footprint=np.asarray(xy)+offset_M[:2]+range_m*direction_M[:2]
    height=terrain.height(footprint)
    if height is None:
        return None
    return float(height-offset_M[2]-range_m*direction_M[2])


class TerrainNavigation(sysModel.SysModel):
    def __init__(self, initial, terrain, sc_state_msg, imu, star_tracker, altimeter, mu, settings=None):
        super().__init__()
        self.cfg=settings or NavigationSettings()
        self.ModelTag='TerrainNavigationUKF' if self.cfg.filter_enabled else 'TerrainNavigationRaw'
        self.initial=initial.copy()
        self.terrain=terrain
        self.truth=sc_state_msg.addSubscriber()
        self.imu=imu.model.sensorOutMsg.addSubscriber()
        self.star=star_tracker.model.sensorOutMsg.addSubscriber()
        # Example calibration: remove the modeled fixed accelerometer bias.
        self.accel_bias=np.asarray(imu.model.senTransBias).reshape(3)
        self.altimeter=altimeter
        self.laser_offset_B=np.asarray(getattr(altimeter,'laser_offset_B',(0.,0.,-.3)))
        self.laser_direction_B=np.asarray(getattr(altimeter,'laser_direction_B',(0.,0.,-1.)))
        self.mu=mu
        self.navOutMsg=messaging.NavTransMsg()
        self.measurement=messaging.NavTransMsg()
        self.epoch=messaging.EphemerisMsg().write(messaging.EphemerisMsgPayload())
        self.filter=smallBodyNavUKF.SmallBodyNavUKF()
        self.filter.mu_ast=mu
        self.filter.alpha=1.
        self.filter.beta=2.
        self.filter.kappa=0.
        self.filter.navTransInMsg.subscribeTo(self.measurement)
        self.filter.asteroidEphemerisInMsg.subscribeTo(self.epoch)
        self.filter.SelfInit()
        self.healthy=True
        self.last_s=None
        self.history=[]
        self.command_source=None
        self.attitude_navigation=None
        self.Reset(0)

    def Reset(self,now):
        self.noise=GaussMarkovXY(self.cfg.xy_sigma_m,self.cfg.xy_correlation_s,self.cfg.seed)
        self.range_rng=np.random.default_rng(self.cfg.seed+1)
        self.filter.x_hat_k=np.r_[self.initial.position,self.initial.velocity,np.zeros(3)].tolist()
        self.filter.P_k=np.diag([1.]*3+[.01]*3+[.001]*3).tolist()
        self.filter.P_proc=np.eye(9).tolist()
        self.filter.R_meas=np.eye(3).tolist()
        self.filter.Reset(now)
        self.raw_state=np.r_[self.initial.position,self.initial.velocity,np.zeros(3)]
        self.previous_fix=None
        self.last_s=None
        self.healthy=True
        self.history.clear()
        payload=messaging.NavTransMsgPayload()
        payload.timeTag=now*1e-9
        payload.r_BN_N=self.initial.position.tolist()
        payload.v_BN_N=self.initial.velocity.tolist()
        self.navOutMsg.write(payload,now,self.moduleID)

    def UpdateState(self,now):
        t=now*1e-9
        if not self.truth.isWritten() or (self.last_s is not None and t-self.last_s < self.cfg.period_s-1e-8):
            return
        dt=0. if self.last_s is None else t-self.last_s
        truth=self.truth()
        # Only the two authorized horizontal coordinates enter the measurement.
        xy=self.terrain.to_map(np.asarray(truth.r_BN_N))[:2]+self.noise.sample(dt)
        c_NB=self.initial.dcm_NB
        if self.attitude_navigation is not None:
            if not self.attitude_navigation.healthy:
                self.healthy=False
                return
            attitude=self.attitude_navigation.navOutMsg.read()
            c_NB=RigidBodyKinematics.MRP2C(attitude.sigma_BN).T
        elif self.star.isWritten():
            c_NB=RigidBodyKinematics.EP2C(self.star().qInrtl2Case).T
        x=(np.asarray(self.filter.x_hat_k).reshape(9) if self.cfg.filter_enabled
           else self.raw_state.copy())
        p=np.asarray(self.filter.P_k).reshape(9,9)
        if self.cfg.filter_enabled and self.command_source is not None:
            controller=self.command_source
            force_B=np.zeros(3)
            mass=controller.initial_state.mass
            if controller.fuelInMsg.isWritten():
                mass=controller.settings.dry_mass+max(0.,controller.fuelInMsg().fuelMass)
            if controller.status in ('descending','free_fall') and controller.last_command is not None:
                force_B=controller.last_command.force_B
            # Known actuator command is a model input, never a truth-position
            # measurement. Anchor acceleration during the unobserved coast;
            # IMU alone double-integrates noise before the map becomes visible.
            model_acc=c_NB@force_B/mass
            model_sigma=.001+.01*np.linalg.norm(model_acc)
            # After contact, external ground loads invalidate this free-flight model.
            if (controller.status in ('coast','descending','free_fall')
                    and not getattr(self.altimeter,'collision',False)):
                r_model=np.eye(3)*model_sigma**2
                k=np.linalg.solve(p[6:,6:]+r_model,p[:,6:].T).T
                x=x+k@(model_acc-x[6:])
                ikh=np.eye(9);ikh[:,6:]-=k
                p=ikh@p@ikh.T+k@r_model@k.T
        if self.imu.isWritten():
            acc_N=c_NB @ (np.asarray(self.imu().AccelPlatform)-self.accel_bias)
            if self.cfg.filter_enabled:
                ra=np.eye(3)*self.cfg.accel_sigma_m_s2**2
                k=np.linalg.solve(p[6:,6:]+ra,p[:,6:].T).T
                x=x+k@(acc_N-x[6:])
                ikh=np.eye(9);ikh[:,6:]-=k
                p=ikh@p@ikh.T+k@ra@k.T
            else:
                x[6:]=acc_N  # calibrated accelerometer, no Kalman gain or smoothing
        if self.cfg.filter_enabled:
            x,p=predict_sigma_points(x,p,dt,self.mu)
        else:
            # One deterministic state, not an unscented distribution. Unobserved
            # Z/velocity are dead-reckoned; spacecraft truth Z is never read.
            def derivative(state):
                return np.r_[state[3:6],-self.mu*state[:3]/np.linalg.norm(state[:3])**3+state[6:],np.zeros(3)]
            k1=derivative(x);k2=derivative(x+.5*dt*k1)
            k3=derivative(x+.5*dt*k2);k4=derivative(x+dt*k3)
            x=x+dt/6*(k1+2*k2+2*k3+k4)
        self.filter.x_hat_k=x.tolist()
        self.filter.P_k=(.5*(p+p.T)+np.eye(9)*1e-12).tolist()
        self.filter.P_proc=np.diag([1e-6]*3+[1e-5]*3+[self.cfg.accel_sigma_m_s2**2*dt]*3).tolist()
        predicted_z=self.terrain.to_map(x[:3])[2]
        z=predicted_z
        z_variance=1e12  # no altitude observation: essentially zero vertical information
        raw=self.altimeter.altitude
        down=float((MAP_TO_N.T @ c_NB @ self.laser_direction_B)[2])
        reason='accepted'
        if not getattr(self.altimeter,'range_valid',False) or not np.isfinite(raw) or raw < 0:
            reason='no_return'
        elif abs(t-getattr(self.altimeter,'range_time_s',-np.inf))>2*self.cfg.period_s:
            reason='stale'
        elif raw>self.cfg.max_range_m:
            reason='over_range'
        elif down>-self.cfg.minimum_down_cosine:
            reason='beam_not_downward'
        valid=reason=='accepted'
        measured=np.nan
        observed=np.nan
        if valid:
            measured=raw+self.range_rng.normal(0,self.cfg.range_sigma_m)
            observed=position_from_range(self.terrain,xy,measured,c_NB,self.laser_offset_B,self.laser_direction_B)
            valid=measured>=0 and observed is not None and np.isfinite(observed)
            if not valid:
                reason='negative_noisy_range' if measured<0 else 'footprint_unmapped'
                observed=np.nan
            if valid:
                z=observed
                # Include range, attitude/beam-footprint and map-location uncertainty.
                z_variance=(self.cfg.range_sigma_m**2+(raw*self.cfg.attitude_sigma_rad)**2
                            +self.cfg.xy_sigma_m**2+.05**2)
        r_M=np.r_[xy,z]
        r_N=self.terrain.to_inertial(r_M)
        measurement=messaging.NavTransMsgPayload()
        measurement.timeTag=t
        measurement.r_BN_N=r_N.tolist()
        self.measurement.write(measurement,now,self.moduleID)
        self.filter.R_meas=(MAP_TO_N@np.diag([self.cfg.xy_sigma_m**2+1e-8]*2+[z_variance])@MAP_TO_N.T).tolist()
        # Distribution already propagated with RK4 above. The native module
        # performs its unscented measurement update without a second time step.
        if self.cfg.filter_enabled:
            self.filter.UpdateState(0)
            output=self.filter.smallBodyNavUKFOutMsg.read()
            estimated=np.asarray(output.state).reshape(9)
            covariance=np.asarray(output.covar).reshape(9,9)
            self.healthy=bool(np.all(np.isfinite(estimated)) and np.all(np.isfinite(covariance))
                              and np.linalg.eigvalsh(.5*(covariance+covariance.T)).min()>-1e-7)
            sigma=np.sqrt(np.maximum(0,np.diag(covariance)[:3]))



        else:
            estimated=x.copy()
            estimated[:3]=r_N  # direct noisy XY and accepted laser Z
            velocity_M=MAP_TO_N.T@estimated[3:6]
            if self.previous_fix is not None and dt>0:
                previous_position,previous_valid=self.previous_fix
                velocity_M[:2]=(r_M[:2]-previous_position[:2])/dt
                if valid and previous_valid:
                    velocity_M[2]=(r_M[2]-previous_position[2])/dt
            estimated[3:6]=MAP_TO_N@velocity_M
            self.raw_state=estimated.copy()
            self.previous_fix=(r_M.copy(),valid)
            self.healthy=bool(np.all(np.isfinite(estimated)))
            sigma=np.full(3,np.nan)  # bypass has no estimated covariance
        if self.healthy:
            payload=messaging.NavTransMsgPayload()
            payload.timeTag=t
            payload.r_BN_N=estimated[:3].tolist()
            payload.v_BN_N=estimated[3:6].tolist()
            self.navOutMsg.write(payload,now,self.moduleID)
        self.last_s=t
        self.history.append(dict(time_s=t,range_valid=bool(valid),healthy=self.healthy,
                                 filter_enabled=self.cfg.filter_enabled,range_status=reason,
                                 raw_slant_range_m=float(raw),used_slant_range_m=   float(measured) if valid else np.nan,
                                 laser_observed_map_z_m=float(observed) if valid else np.nan,
                                 predicted_map_z_m=float(predicted_z),
                                 estimated_map_z_m=float(self.terrain.to_map(estimated[:3])[2]),
                                 measured_map_xy_m=xy.copy(),beam_down_cosine=-down,
                                 position_N=estimated[:3].copy(),velocity_N=estimated[3:6].copy(),
                                 position_sigma_m=sigma))
