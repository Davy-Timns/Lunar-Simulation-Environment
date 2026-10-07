"""IMU gyro attitude propagation with star-tracker drift correction; no truth input."""
import numpy as np
from scipy.spatial.transform import Rotation
from Basilisk.architecture import messaging, sysModel
from Basilisk.utilities import RigidBodyKinematics


class AttitudeNavigation(sysModel.SysModel):
    def __init__(self, initial_dcm_NB, imu, star_tracker, period_s=.01, correction_time_s=1.):
        super().__init__()
        if min(period_s, correction_time_s) <= 0 or not np.all(np.isfinite([period_s, correction_time_s])):
            raise ValueError('Attitude sample/correction periods must be finite and positive')
        self.ModelTag = 'IMUAttitudeNavigation'
        self.initial_dcm = np.asarray(initial_dcm_NB, dtype=float).reshape(3, 3).copy()
        self.imu = imu.model.sensorOutMsg.addSubscriber()
        self.star = star_tracker.model.sensorOutMsg.addSubscriber()
        self.gyro_bias = np.asarray(imu.model.senRotBias, dtype=float).reshape(3)
        self.dcm_PB = np.asarray(imu.model.dcm_PB, dtype=float).reshape(3, 3)
        self.dcm_CB = np.asarray(star_tracker.model.dcm_CB, dtype=float).reshape(3, 3)
        self.period_s = period_s
        self.correction_time_s = correction_time_s
        self.navOutMsg = messaging.NavAttMsg()
        self.Reset(0)

    def Reset(self, now):
        self.dcm_NB = self.initial_dcm.copy()
        self.omega_B = np.zeros(3)
        self.last_s = None
        self.last_star_ns = None
        self.initialized = False
        self.healthy = False
        self.history = []
        self._publish(now)

    def _publish(self, now):
        payload = messaging.NavAttMsgPayload()
        payload.timeTag = now * 1e-9
        payload.sigma_BN = RigidBodyKinematics.C2MRP(self.dcm_NB.T).tolist()
        payload.omega_BN_B = self.omega_B.tolist()
        self.navOutMsg.write(payload, now, self.moduleID)

    def UpdateState(self, now):
        t = now * 1e-9
        if not self.imu.isWritten() or t - self.imu.timeWritten() * 1e-9 > 2 * self.period_s + 1e-9:
            self.healthy = False
            return
        rate = self.dcm_PB.T @ (np.asarray(self.imu().AngVelPlatform) - self.gyro_bias)
        if not np.all(np.isfinite(rate)):
            self.healthy = False
            return
        dt = 0. if self.last_s is None else max(0., t - self.last_s)
        if dt:
            mean_rate = .5 * (self.omega_B + rate)
            self.dcm_NB = self.dcm_NB @ Rotation.from_rotvec(dt * mean_rate).as_matrix()
        self.omega_B = rate
        corrected = False
        if self.star.isWritten() and self.star.timeWritten() != self.last_star_ns:
            q = np.asarray(self.star().qInrtl2Case, dtype=float)
            if np.all(np.isfinite(q)) and np.linalg.norm(q) > 1e-12 and t - self.star.timeWritten() * 1e-9 <= .2:
                measured = RigidBodyKinematics.EP2C((q / np.linalg.norm(q)).tolist()).T @ self.dcm_CB
                if not self.initialized:
                    self.dcm_NB = measured
                else:
                    error = Rotation.from_matrix(self.dcm_NB.T @ measured).as_rotvec()
                    gain = -np.expm1(-max(dt, self.period_s) / self.correction_time_s)
                    self.dcm_NB = self.dcm_NB @ Rotation.from_rotvec(gain * error).as_matrix()
                self.last_star_ns = self.star.timeWritten()
                self.initialized = True
                corrected = True
        # Gyros give attitude changes; an absolute sensor fix is needed first.
        self.healthy = bool(self.initialized and np.all(np.isfinite(self.dcm_NB)))
        self.last_s = t
        if self.healthy:
            self._publish(now)
        self.history.append(dict(time_s=t, omega_B_rad_s=rate.copy(),
                                 dcm_NB=self.dcm_NB.copy(), star_corrected=corrected, healthy=self.healthy))
