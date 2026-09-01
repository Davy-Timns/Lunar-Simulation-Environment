# StarTracker.py
from Basilisk.simulation import starTracker
from Basilisk.architecture.messaging.STSensorMsgPayload import STSensorMsgPayload

#This startracker utilizes the Rocketlab ST-HV Star Tracker because it has a datasheet available, its not much but it is a baseline... https://rocketlabcorp.com/assets/Uploads/ST-HV-Datasheet-v3.4.pdf
class StarTracker:
    def __init__(self, name="StarTracker"):
        self.name = name                                #Name of the sensor
        self.model = starTracker.StarTracker()          #Star tracker model init
        self.model.ModelTag = name                      #Tag the model with the name as well

        self.model.dcm_CB = [                           #DCM of body to case
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        self.model.mrpErrors = [0.000034, 0.000034, 0.00034] #Sensor attitude error (rads) added to MRP output of quaternion
                                                              #ST-HV Star Tracker has a near-boresight accuracy of 7 arcseconds, ie 0.000034 rads
                                                              #ST-HV Star Tracker has an on-boresight accuracy of 70 arcseconds, ie 0.00034 rads
                                                              #Because dcm_CB shows we are pointing along the spacecraft's Z axis, the last MRP error (z axis) is 70 arcseconds
        self.model.PMatrix = [                          #P Matrix is the noise covariance and used for noise propagation in the Gauss-Markov model.
            [0.001, 0.0, 0.0],
            [0.0, 0.001, 0.0],
            [0.0, 0.0, 0.001]
        ]

        #self.model.setWalkBounds([0.034, 0.034, 0.34]) #maximum accumulated error magnitude


        #self.model.setAMatrix(self.model.getAMatrix())  #Set MRP shadow set
        self.recorder = None                            #Init recorder var
        self.fields = STSensorMsgPayload.__fields__()   #Retrieves all fields in the ST output message

    def connect_state(self, sc_state_msg):
        """
        Connect state message to star tracker
        :param sc_state_msg: State message
        :return:
        """
        self.model.scStateInMsg.subscribeTo(sc_state_msg)

    def attach_recorder(self, sampling_ns):
        """
        Attach recorder/logger to star tracker
        :param sampling_ns: Sampling rate freq
        :return:
        """
        self.recorder = self.model.sensorOutMsg.recorder(sampling_ns)
        return self.recorder

    def configure(self, cfg):
        """
        Configure star tracker to replace default config
        :param cfg: New config
        :return:
        """
        for key, val in cfg.items():
            setattr(self.model, key, val)

    def process(self, env):
        """Black-box processing step, empty for rn"""
        pass

    def output(self, index):
        """Return all star tracker fields at a timestep"""
        return {name: getattr(self.recorder, name)[index] for name in self.fields}
