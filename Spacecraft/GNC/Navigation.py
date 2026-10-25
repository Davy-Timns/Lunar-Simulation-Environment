"""Terrain geometry, sensor calls and joint position/attitude navigation.

Map coordinates are local X/Y horizontal, Z up, with origin at the south-pole
reference surface. Unmapped locations return None, never a fabricated height.
"""
from pathlib import Path
import hashlib
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from Basilisk.architecture import messaging, sysModel
from Basilisk.utilities import RigidBodyKinematics
from Spacecraft.Mujoco.FrameTransforms import R_MUJOCO_TO_BSK, TERRAIN_POSITION_BSK

MAP_TO_N = np.array([[1.,0.,0.], [0.,0.,-1.], [0.,1.,0.]])
BSK_TO_UNITY = np.array([[0.,1.,0.], [0.,0.,1.], [-1.,0.,0.]])


class TerrainMap:
    up_N = MAP_TO_N[:, 2]
    def __init__(self, altimeter):
        self.model = altimeter.model
        # Separate data prevents map queries from changing sensor/contact state.
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model,self.data)
        self.geom = altimeter.terrain_geom_id
        self.body_exclude = altimeter.spacecraft_body_id
        if self.geom < 0:
            raise ValueError('Terrain map requires the lunar terrain mesh')
        mesh = self.model.geom_dataid[self.geom]
        start = self.model.mesh_vertadr[mesh]
        n = self.model.mesh_vertnum[mesh]
        vertices = np.array(self.model.mesh_vert[start:start+n],dtype=float)
        self.vertices_N = (vertices @ self.data.geom_xmat[self.geom].reshape(3,3).T + self.data.geom_xpos[self.geom]) @ R_MUJOCO_TO_BSK.T
        vertices_M = self.to_map(self.vertices_N)
        self.minimum = vertices_M.min(axis=0)
        self.maximum = vertices_M.max(axis=0)
        start = self.model.mesh_faceadr[mesh]
        n = self.model.mesh_facenum[mesh]
        self.faces = np.array(self.model.mesh_face[start:start+n],dtype=int)
        # Preload an XY spatial index of the mesh. All later elevation queries
        # interpolate these fixed triangles; they do not fire another sensor
        # ray or read the spacecraft pose. Preserve the actual mesh topology.
        triangles=vertices_M[self.faces]
        self._triangle_origin=triangles[:,0]
        self._triangle_u=triangles[:,1]-triangles[:,0]
        self._triangle_v=triangles[:,2]-triangles[:,0]
        self._det=(self._triangle_u[:,0]*self._triangle_v[:,1]
                   -self._triangle_u[:,1]*self._triangle_v[:,0])
        self._cell_m=50.
        self._cells={}
        lower=np.floor((triangles[:,:,:2].min(axis=1)-self.minimum[:2])/self._cell_m).astype(int)
        upper=np.floor((triangles[:,:,:2].max(axis=1)-self.minimum[:2])/self._cell_m).astype(int)
        for i in np.flatnonzero(np.abs(self._det)>1e-12):
            for ix in range(lower[i,0],upper[i,0]+1):
                for iy in range(lower[i,1],upper[i,1]+1):
                    self._cells.setdefault((ix,iy),[]).append(i)
        self._cells={key:np.asarray(value,dtype=int) for key,value in self._cells.items()}

    @staticmethod
    def to_map(r_N):
        return (np.asarray(r_N)-TERRAIN_POSITION_BSK) @ MAP_TO_N

    @staticmethod
    def to_inertial(r_M):
        """Local map metres -> Moon-centred inertial metres, for this static map.

        Column-vector form: r_N = origin_N + MAP_TO_N @ r_M. NumPy uses
        row vectors here so a batch of positions also works. This is a fixed
        rigid transform, not latitude/longitude conversion or Moon rotation.
        Local +X -> inertial +X; local +Y -> inertial +Z; local +Z -> inertial -Y.
        """
        return np.asarray(r_M) @ MAP_TO_N.T + TERRAIN_POSITION_BSK

    def height(self, xy):
        """Preloaded DEM elevation at map XY, not vehicle height or a range.

        Barycentric interpolation on the compiled collision-mesh triangles.
        The uppermost surface is used if multiple triangles cover the same XY.
        """
        xy = np.asarray(xy,dtype=float)
        if np.any(xy < self.minimum[:2]) or np.any(xy > self.maximum[:2]):
            return None
        cell=tuple(np.floor((xy-self.minimum[:2])/self._cell_m).astype(int))
        ids=self._cells.get(cell)
        if ids is None:
            return None
        delta=xy-self._triangle_origin[ids,:2]
        u=self._triangle_u[ids];v=self._triangle_v[ids]
        a=(delta[:,0]*v[:,1]-delta[:,1]*v[:,0])/self._det[ids]
        b=(u[:,0]*delta[:,1]-u[:,1]*delta[:,0])/self._det[ids]
        inside=(a>=-1e-9)&(b>=-1e-9)&(a+b<=1+1e-9)
        if not inside.any():
            return None
        heights=self._triangle_origin[ids,2]+a*u[:,2]+b*v[:,2]
        return float(np.max(heights[inside]))

    def clearance(self, r_N):
        """Map-relative clearance of a supplied position.

        Planning/assessment use simulated positions; the engine cutoff uses
        estimated foot positions. Navigation Z comes from laser reconstruction
        or inertial propagation, never from this function.
        """
        p = self.to_map(r_N)
        h = self.height(p[:2])
        return None if h is None else float(p[2]-h)

    def assess_site(self, xy, radius_m=2., spacing_m=1., max_slope_deg=10., max_roughness_m=.2):
        samples=[]
        for dx in np.arange(-radius_m,radius_m+.5*spacing_m,spacing_m):
            for dy in np.arange(-radius_m,radius_m+.5*spacing_m,spacing_m):
                h=self.height(np.asarray(xy)+[dx,dy])
                if h is None:
                    return dict(mapped=False,acceptable=False,reason='outside map coverage')
                samples.append([dx,dy,h])
        points=np.asarray(samples)
        design=np.column_stack([points[:,:2],np.ones(len(points))])
        fit=np.linalg.lstsq(design,points[:,2],rcond=None)[0]
        slope=float(np.rad2deg(np.arctan(np.linalg.norm(fit[:2]))))
        roughness=float(np.max(np.abs(points[:,2]-design@fit)))
        return dict(mapped=True,acceptable=slope<=max_slope_deg and roughness<=max_roughness_m,
                    slope_deg=slope,roughness_m=roughness,height_m=float(fit[2]),
                    maximum_height_m=float(points[:,2].max()),footprint_radius_m=radius_m)

    def find_site(self, preferred_xy, search_radius_m=150., spacing_m=10., **criteria):
        candidates=sorted((dx*dx+dy*dy,dx,dy)
                          for dx in np.arange(-search_radius_m,search_radius_m+.1,spacing_m)
                          for dy in np.arange(-search_radius_m,search_radius_m+.1,spacing_m)
                          if dx*dx+dy*dy <= search_radius_m**2)
        for _,dx,dy in candidates:
            xy=np.asarray(preferred_xy)+[dx,dy]
            result=self.assess_site(xy,**criteria)
            if result['acceptable']:
                return xy,result
        raise ValueError('No mapped site meets the configured slope/roughness/footprint criteria')

    def export_vizard_obj(self, path, versioned=False):
        """Bake the collision vertices into Vizard's Unity local coordinates.

        Use rotation=[0,0,0], scale=[1,1,1], and the existing terrain anchor.
        Vizard's OBJ custom-model path consumes raw vertices in Unity axes.
        """
        path=Path(path)
        vertices=(self.vertices_N-TERRAIN_POSITION_BSK) @ BSK_TO_UNITY.T
        signature=hashlib.sha256(vertices.tobytes()+self.faces.tobytes()).hexdigest()
        if versioned:
            # Old recordings must keep their original geometry when the
            # terrain's origin or collision resolution changes.
            path = path.with_name(f'{path.stem}_{signature[:16]}{path.suffix}')
        if path.exists() and path.with_suffix('.sha256').exists() and path.with_suffix('.sha256').read_text()==signature:
            return str(path.resolve())
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w') as stream:
            stream.write('# Collision-aligned terrain; Unity local coordinates, metres\n')
            for v in vertices:
                stream.write('v %.9f %.9f %.9f\n'%tuple(v))
            # Right-handed -> left-handed coordinate conversion reverses winding.
            for a,b,c in self.faces:
                stream.write(f'f {a+1} {c+1} {b+1}\n')
        path.with_suffix('.sha256').write_text(signature)
        return str(path.resolve())


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

class NavigationSettings:
    def __init__(self,filter_enabled=True,period_s=0.1,xy_sigma_m=0.25,xy_correlation_s=10.0,range_sigma_m=0.05,max_range_m=5000.0,minimum_down_cosine=0.5,attitude_sigma_rad=0.004,accel_sigma_m_s2=0.09,seed=42,):
        self.filter_enabled = filter_enabled
        self.period_s = period_s
        self.xy_sigma_m = xy_sigma_m
        self.xy_correlation_s = xy_correlation_s
        self.range_sigma_m = range_sigma_m
        self.max_range_m = max_range_m
        self.minimum_down_cosine = minimum_down_cosine
        self.attitude_sigma_rad = attitude_sigma_rad
        self.accel_sigma_m_s2 = accel_sigma_m_s2
        self.seed = seed
        if not isinstance(self.filter_enabled, bool):
            raise ValueError('filter_enabled must be a boolean fixed at simulation startup')
        if not np.all(np.isfinite(list(vars(self).values()))):
            raise ValueError('Navigation settings must be finite')
        if min(self.period_s,self.xy_correlation_s,self.max_range_m,self.accel_sigma_m_s2) <= 0:
            raise ValueError('Navigation periods, range and acceleration sigma must be positive')
        if min(self.xy_sigma_m,self.range_sigma_m,self.attitude_sigma_rad) < 0 or not 0 < self.minimum_down_cosine <= 1:
            raise ValueError('Invalid sensor noise or laser down-cone settings')



def position_from_range(terrain, xy, range_m, c_NB, offset_B=(0.,0.,-.3), direction_B=(0.,0.,-1.)):
    """Convert SLANT range to COM map height, including beam footprint/offset."""
    offset_M=MAP_TO_N.T @ c_NB @ np.asarray(offset_B)
    direction_M=MAP_TO_N.T @ c_NB @ np.asarray(direction_B)
    footprint=np.asarray(xy)+offset_M[:2]+range_m*direction_M[:2]
    height=terrain.height(footprint)
    if height is None:
        return None
    return float(height-offset_M[2]-range_m*direction_M[2])




class UKF:
    """Joint manifold UKF with 15 tangent coordinates and one covariance.

    State: inertial position/velocity/non-gravitational acceleration, body-to-
    inertial rotation, and body angular rate. Attitude errors use right-local
    rotation vectors, never Euler angles. Sigma weights are alpha=1, beta=2,
    kappa=0. Gyro, accelerometer, star and terrain observations all update P.
    Fixed modeled sensor biases are calibrated outside this filter; estimating
    unknown bias drift would require additional states and a drift model.
    """
    dimension = 15

    def __init__(self, initial, mu):
        self.mu = float(mu)
        if not np.isfinite(mu) or mu <= 0:
            raise ValueError('UKF requires positive finite gravitational parameter')
        self.reset(initial)

    def reset(self, initial):
        self.translation = np.r_[initial.position, initial.velocity, np.zeros(3)]
        self.rotation = np.asarray(initial.dcm_NB, dtype=float).copy()
        self.omega = np.asarray(initial.omega_B, dtype=float).copy()
        self.P = np.diag([1.]*3 + [.01]*3 + [.001]*3 + [.004**2]*3 + [.004**2]*3)

    @staticmethod
    def _positive(covariance):
        covariance = .5 * (covariance + covariance.T)
        if not np.isfinite(covariance).all():
            raise ValueError('Non-finite UKF covariance')
        # Only roundoff-scale diagonal regularization is permitted. A genuinely
        # indefinite covariance causes a navigation fault instead of hiding it.
        for jitter in (0., 1e-12, 1e-10, 1e-8):
            candidate = covariance + jitter * np.eye(len(covariance))
            try:
                np.linalg.cholesky(candidate)
                return candidate
            except np.linalg.LinAlgError:
                continue
        raise ValueError('UKF covariance is not positive definite')

    def sigma_points(self):
        self.P = self._positive(self.P)
        root = np.linalg.cholesky(self.P) * np.sqrt(self.dimension)
        delta = np.vstack((np.zeros(self.dimension), root.T, -root.T))
        return (self.translation + delta[:, :9],
                self.rotation @ Rotation.from_rotvec(delta[:, 9:12]).as_matrix(),
                self.omega + delta[:, 12:])

    @staticmethod
    def _covariance(a, b):
        return a[1:].T @ b[1:] / 30. + 2 * np.outer(a[0], b[0])

    def _errors(self, points):
        translation, rotations, rates = points
        angles = Rotation.from_matrix(self.rotation.T @ rotations).as_rotvec()
        return np.column_stack((translation - self.translation, angles, rates - self.omega))

    def predict(self, dt, accel_sigma=.09, gyro_sigma=.004):
        if not np.isfinite(dt) or dt < 0:
            raise ValueError('Prediction interval must be finite and nonnegative')
        if not dt:
            return
        x, rotations, rates = self.sigma_points()
        def derivative(states):
            radius = np.linalg.norm(states[:, :3], axis=1)
            if np.any(radius <= 0):
                raise ValueError('Sigma point at the gravity origin')
            result = np.zeros_like(states)
            result[:, :3] = states[:, 3:6]
            result[:, 3:6] = -self.mu * states[:, :3] / radius[:, None]**3 + states[:, 6:9]
            return result
        k1 = derivative(x)
        k2 = derivative(x + .5 * dt * k1)
        k3 = derivative(x + .5 * dt * k2)
        k4 = derivative(x + dt * k3)
        x = x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        rotations = rotations @ Rotation.from_rotvec(dt * rates).as_matrix()
        # Compute means relative to the central point to avoid cancellation of
        # metre-scale uncertainty on Moon-centred million-metre coordinates.
        self.translation = x[0] + np.mean(x[1:] - x[0], axis=0)
        self.omega = rates[0] + np.mean(rates[1:] - rates[0], axis=0)
        self.rotation = rotations[0].copy()
        for _ in range(5):
            correction = np.mean(Rotation.from_matrix(self.rotation.T @ rotations[1:]).as_rotvec(), axis=0)
            self.rotation = self.rotation @ Rotation.from_rotvec(correction).as_matrix()
            if np.linalg.norm(correction) < 1e-12:
                break
        error = self._errors((x, rotations, rates))
        process = np.diag([1e-6]*3 + [1e-5]*3 + [accel_sigma**2]*3
                          + [1e-10]*3 + [gyro_sigma**2]*3) * dt
        self.P = self._positive(self._covariance(error, error) + process)

    @staticmethod
    def _right_jacobian(angle):
        """Transport attitude covariance when the local rotation origin moves."""
        x, y, z = angle
        skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
        norm = np.linalg.norm(angle)
        if norm < 1e-6:
            return np.eye(3) - .5*skew + skew@skew/6
        return (np.eye(3) - (1-np.cos(norm))/norm**2*skew
                + (norm-np.sin(norm))/norm**3*(skew@skew))

    def _apply(self, correction, covariance):
        self.translation += correction[:9]
        self.rotation = self.rotation @ Rotation.from_rotvec(correction[9:12]).as_matrix()
        self.omega += correction[12:]
        transport = np.eye(self.dimension)
        transport[9:12, 9:12] = self._right_jacobian(correction[9:12])
        self.P = self._positive(transport @ covariance @ transport.T)

    def update(self, observation, observation_function, noise):
        """Unscented update of all states, including translation/attitude coupling."""
        points = self.sigma_points()
        values = np.asarray(observation_function(*points), dtype=float)
        observation = np.asarray(observation, dtype=float).reshape(-1)
        noise = np.asarray(noise, dtype=float)
        if values.shape != (31, len(observation)) or noise.shape != (len(observation),)*2:
            raise ValueError('Invalid UKF measurement dimensions')
        if not np.isfinite(values).all() or not np.isfinite(observation).all():
            raise ValueError('Non-finite UKF observation')
        mean = values[0] + np.mean(values[1:] - values[0], axis=0)
        residuals = values - mean
        innovation = self._positive(self._covariance(residuals, residuals) + noise)
        cross = self._covariance(self._errors(points), residuals)
        gain = np.linalg.solve(innovation, cross.T).T
        self._apply(gain @ (observation - mean), self.P - gain @ innovation @ gain.T)

    def update_rate(self, rate, sigma=.004):
        # Angular-rate observation is linear in the joint tangent coordinates.
        # Use its exact unscented moments, avoiding extra rotation conversions.
        noise = np.eye(3) * sigma**2
        gain = np.linalg.solve(self.P[12:, 12:] + noise, self.P[:, 12:].T).T
        residual = np.asarray(rate) - self.omega
        ikh = np.eye(15)
        ikh[:, 12:] -= gain
        self._apply(gain @ residual, ikh @ self.P @ ikh.T + gain @ noise @ gain.T)

    def update_star(self, rotation, sigma=.004):
        reference = self.rotation.copy()
        measured = Rotation.from_matrix(reference.T @ rotation).as_rotvec()
        self.update(measured, lambda x, r, w: Rotation.from_matrix(reference.T @ r).as_rotvec(),
                    np.eye(3) * sigma**2)

    def update_acceleration(self, acceleration_B, sigma):
        self.update(acceleration_B,
                    lambda x, r, w: np.einsum('nji,nj->ni', r, x[:, 6:9]),
                    np.eye(3) * sigma**2)


class TerrainNavigation(sysModel.SysModel):
    """Read sensor messages, gate terrain observations, publish joint estimates.

    The only simulated truth measurement is the explicitly noisy map X/Y fix.
    Truth Z, velocity, attitude and body rate never enter the UKF. The initial
    prior is supplied by mission setup. Raw mode bypasses every Kalman update.
    """
    def __init__(self, initial, terrain, sc_state_msg, imu, star_tracker, altimeter,
                 mu, settings=None, sensor_period_s=.01):
        super().__init__()
        self.cfg = settings or NavigationSettings()
        if not np.isfinite(sensor_period_s) or sensor_period_s <= 0:
            raise ValueError('Sensor period must be positive and finite')
        self.sensor_period_s = float(sensor_period_s)
        self.ModelTag = 'JointTerrainNavigationUKF' if self.cfg.filter_enabled else 'TerrainNavigationRaw'
        self.initial = initial.copy()
        self.terrain = terrain
        self.truth = sc_state_msg.addSubscriber()
        self.imu = imu.model.sensorOutMsg.addSubscriber()
        self.star = star_tracker.model.sensorOutMsg.addSubscriber()
        self.accel_bias = np.asarray(imu.model.senTransBias).reshape(3)
        self.gyro_bias = np.asarray(imu.model.senRotBias).reshape(3)
        self.dcm_PB = np.asarray(imu.model.dcm_PB).reshape(3, 3)
        self.dcm_CB = np.asarray(star_tracker.model.dcm_CB).reshape(3, 3)
        self.altimeter = altimeter
        self.laser_offset_B = np.asarray(getattr(altimeter, 'laser_offset_B', (0., 0., -.3)))
        self.laser_direction_B = np.asarray(getattr(altimeter, 'laser_direction_B', (0., 0., -1.)))
        self.mu = mu
        self.navOutMsg = messaging.NavTransMsg()
        self.attitudeOutMsg = messaging.NavAttMsg()
        self.ukf = UKF(initial, mu)
        self.command_source = None
        self.Reset(0)

    def Reset(self, now):
        self.ukf.reset(self.initial)
        self.noise = GaussMarkovXY(self.cfg.xy_sigma_m, self.cfg.xy_correlation_s, self.cfg.seed)
        self.range_rng = np.random.default_rng(self.cfg.seed+1)
        self.last_s = None
        self.last_fix_s = None
        self.last_star_ns = None
        self.last_imu_ns = None
        self.absolute_attitude = False
        self.healthy = False
        self.fault_reason = None
        self.history = []
        self.attitude_history = []
        self.previous_fix = None
        self._publish(now)

    def _publish(self, now):
        payload = messaging.NavTransMsgPayload()
        payload.timeTag = now*1e-9
        payload.r_BN_N = self.ukf.translation[:3].tolist()
        payload.v_BN_N = self.ukf.translation[3:6].tolist()
        self.navOutMsg.write(payload, now, self.moduleID)
        attitude = messaging.NavAttMsgPayload()
        attitude.timeTag = now*1e-9
        attitude.sigma_BN = RigidBodyKinematics.C2MRP(self.ukf.rotation.T).tolist()
        attitude.omega_BN_B = self.ukf.omega.tolist()
        self.attitudeOutMsg.write(attitude, now, self.moduleID)

    def _star_rotation(self, t):
        if not self.star.isWritten() or self.star.timeWritten() == self.last_star_ns:
            return None
        q = np.asarray(self.star().qInrtl2Case)
        if (t-self.star.timeWritten()*1e-9 > .2 or not np.isfinite(q).all()
                or np.linalg.norm(q) < 1e-12):
            return None
        self.last_star_ns = self.star.timeWritten()
        return RigidBodyKinematics.EP2C((q/np.linalg.norm(q)).tolist()).T @ self.dcm_CB

    def _command_acceleration(self):
        controller = self.command_source
        if controller is None or getattr(self.altimeter, 'collision', False):
            return None
        if controller.status not in ('coast', 'descending', 'free_fall'):
            return None
        mass = controller.initial_state.mass
        if controller.fuelInMsg.isWritten():
            mass = controller.settings.dry_mass + max(0., controller.fuelInMsg().fuelMass)
        force = np.zeros(3)
        if controller.status in ('descending', 'free_fall') and controller.last_command is not None:
            force = controller.last_command.force_B
        return force / mass

    def _range_residual(self, x, rotations, rates, measured):
        positions = self.terrain.to_map(x[:, :3])
        offsets = np.einsum('nij,j->ni', rotations, self.laser_offset_B) @ MAP_TO_N
        directions = np.einsum('nij,j->ni', rotations, self.laser_direction_B) @ MAP_TO_N
        footprint = positions[:, :2] + offsets[:, :2] + measured*directions[:, :2]
        heights = [self.terrain.height(point) for point in footprint]
        if any(height is None for height in heights):
            raise LookupError('Laser sigma footprint outside terrain coverage')
        return (positions[:, 2] + offsets[:, 2] + measured*directions[:, 2] - heights)[:, None]

    def _terrain_update(self, now, t, dt):
        # Only these two authorized coordinates are read from the truth message.
        xy = self.terrain.to_map(np.asarray(self.truth().r_BN_N))[:2] + self.noise.sample(dt)
        if self.cfg.filter_enabled:
            self.ukf.update(xy, lambda x, r, w: self.terrain.to_map(x[:, :3])[:, :2],
                            np.eye(2)*(self.cfg.xy_sigma_m**2+1e-8))
        raw = self.altimeter.altitude
        c_NB = self.ukf.rotation
        down = float((MAP_TO_N.T @ c_NB @ self.laser_direction_B)[2])
        reason = 'accepted'
        if not getattr(self.altimeter, 'range_valid', False) or not np.isfinite(raw) or raw < 0:
            reason = 'no_return'
        elif abs(t-getattr(self.altimeter, 'range_time_s', -np.inf)) > 2*self.cfg.period_s:
            reason = 'stale'
        elif raw > self.cfg.max_range_m:
            reason = 'over_range'
        elif down > -self.cfg.minimum_down_cosine:
            reason = 'beam_not_downward'
        valid = reason == 'accepted'
        measured = observed = np.nan
        predicted_z = self.terrain.to_map(self.ukf.translation[:3])[2]
        if valid:
            measured = raw + self.range_rng.normal(0, self.cfg.range_sigma_m)
            observed = position_from_range(self.terrain, xy, measured, c_NB,
                                           self.laser_offset_B, self.laser_direction_B)
            if measured < 0:
                valid, reason = False, 'negative_noisy_range'
            elif observed is None or not np.isfinite(observed):
                valid, reason = False, 'footprint_unmapped'
            elif self.cfg.filter_enabled:
                try:
                    # Implicit slant-range/DEM observation. Each attitude sigma
                    # rotates the beam and shifts its own terrain footprint;
                    # attitude uncertainty is propagated, not added twice to R.
                    self.ukf.update([0.], lambda x, r, w: self._range_residual(x, r, w, measured),
                                    np.array([[self.cfg.range_sigma_m**2*down**2 + .05**2]]))
                except LookupError:
                    valid, reason = False, 'footprint_unmapped'
        if not self.cfg.filter_enabled:
            position_M = np.r_[xy, observed if valid else predicted_z]
            velocity_M = MAP_TO_N.T @ self.ukf.translation[3:6]
            if self.previous_fix is not None and dt > 0:
                previous_position, previous_valid = self.previous_fix
                velocity_M[:2] = (position_M[:2]-previous_position[:2])/dt
                if valid and previous_valid:
                    velocity_M[2] = (position_M[2]-previous_position[2])/dt
            self.ukf.translation[:3] = self.terrain.to_inertial(position_M)
            self.ukf.translation[3:6] = MAP_TO_N @ velocity_M
            self.previous_fix = (position_M.copy(), valid)
        sigma = (np.sqrt(np.maximum(0, np.diag(self.ukf.P)[:3]))
                 if self.cfg.filter_enabled else np.full(3, np.nan))
        self.history.append(dict(time_s=t, range_valid=bool(valid), healthy=True,
            filter_enabled=self.cfg.filter_enabled, range_status=reason,
            raw_slant_range_m=float(raw), used_slant_range_m=float(measured) if valid else np.nan,
            laser_observed_map_z_m=float(observed) if valid else np.nan,
            predicted_map_z_m=float(predicted_z),
            estimated_map_z_m=float(self.terrain.to_map(self.ukf.translation[:3])[2]),
            measured_map_xy_m=xy.copy(), beam_down_cosine=-down,
            position_N=self.ukf.translation[:3].copy(), velocity_N=self.ukf.translation[3:6].copy(),
            position_sigma_m=sigma))

    def UpdateState(self, now):
        t = now*1e-9
        if not self.imu.isWritten() or t-self.imu.timeWritten()*1e-9 > 2*self.sensor_period_s+1e-9:
            self.healthy, self.fault_reason = False, 'stale_imu'
            return
        if self.last_s is not None and t <= self.last_s:
            return
        dt = 0. if self.last_s is None else t-self.last_s
        fix_due = self.last_fix_s is None or t-self.last_fix_s >= self.cfg.period_s-1e-8
        sensor = self.imu()
        rate = self.dcm_PB.T @ (np.asarray(sensor.AngVelPlatform)-self.gyro_bias)
        acceleration = self.dcm_PB.T @ (np.asarray(sensor.AccelPlatform)-self.accel_bias)
        if not np.isfinite(np.r_[rate, acceleration]).all():
            self.healthy, self.fault_reason = False, 'nonfinite_imu'
            return
        try:
            if self.cfg.filter_enabled:
                # Use the gyro as an observation of the joint rate state, then
                # propagate attitude with that estimate. Never overwrite a
                # filtered state directly with a measurement.
                if self.imu.timeWritten() != self.last_imu_ns:
                    self.ukf.update_rate(rate)
                self.ukf.predict(dt, self.cfg.accel_sigma_m_s2)
            else:
                previous_rate = self.ukf.omega.copy()
                self.ukf.rotation = self.ukf.rotation @ Rotation.from_rotvec(.5*dt*(previous_rate+rate)).as_matrix()
                self.ukf.omega = rate.copy()
                self.ukf.translation[6:9] = self.ukf.rotation @ acceleration
                x = self.ukf.translation
                def derivative(state):
                    return np.r_[state[3:6], -self.mu*state[:3]/np.linalg.norm(state[:3])**3+state[6:9], np.zeros(3)]
                k1 = derivative(x); k2 = derivative(x+.5*dt*k1)
                k3 = derivative(x+.5*dt*k2); k4 = derivative(x+dt*k3)
                self.ukf.translation = x+dt/6*(k1+2*k2+2*k3+k4)
            self.last_imu_ns = self.imu.timeWritten()
            corrected = False
            if fix_due or not self.absolute_attitude:
                star = self._star_rotation(t)
                if star is not None:
                    if self.cfg.filter_enabled:
                        self.ukf.update_star(star, max(self.cfg.attitude_sigma_rad, 1e-6))
                    else:
                        self.ukf.rotation = star
                    self.absolute_attitude = True
                    corrected = True
            if not self.absolute_attitude:
                self.healthy, self.fault_reason = False, 'awaiting_star_fix'
                self.last_s = t
                return
            if fix_due:
                if self.cfg.filter_enabled:
                    command = self._command_acceleration()
                    if command is not None:
                        self.ukf.update_acceleration(command, .001+.01*np.linalg.norm(command))
                    self.ukf.update_acceleration(acceleration, self.cfg.accel_sigma_m_s2)
                if self.truth.isWritten():
                    fix_dt = 0. if self.last_fix_s is None else t-self.last_fix_s
                    self._terrain_update(now, t, fix_dt)
                    self.last_fix_s = t
            self.healthy = bool(np.isfinite(self.ukf.translation).all() and np.isfinite(self.ukf.rotation).all())
            self.fault_reason = None if self.healthy else 'nonfinite_state'
            self.last_s = t
            if self.healthy:
                self._publish(now)
            self.attitude_history.append(dict(time_s=t, omega_B_rad_s=self.ukf.omega.copy(),
                dcm_NB=self.ukf.rotation.copy(), star_corrected=corrected, healthy=self.healthy))
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.healthy, self.fault_reason = False, str(exc)
