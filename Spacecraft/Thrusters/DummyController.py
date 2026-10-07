"""Open-loop controller: selected engines remain on at their configured MaxThrust."""

import math

from Basilisk.architecture import messaging, sysModel


class DummyController(sysModel.SysModel):
    """Publish one on-time command per engine on every controller tick.

    Run on the same task as the engines, with controller priority 20,
    engine priority 10, and spacecraft priority 0. Selected engines deliver
    their configured MaxThrust in newtons; this module does not implement PWM,
    throttle, feedback, or actuator dynamics. Use configs without ramps.
    """

    def __init__(self, engines, period_s, enabled_engines=(), name="DummyController"):
        super().__init__()
        self.ModelTag = name
        self.period_s = float(period_s)
        if not math.isfinite(self.period_s) or self.period_s <= 0:
            raise ValueError("period_s must be finite and positive.")

        self.engines = dict(engines)
        if len({id(engine.thruster) for engine in self.engines.values()}) != len(self.engines):
            raise ValueError("Each engine name must refer to a distinct effector.")
        for engine in self.engines.values():
            config = engine.config
            if len(config.ThrusterOnRamp) or len(config.ThrusterOffRamp):
                raise ValueError("DummyController requires thrusters without on/off ramps.")
            if len(config.thrBlowDownCoeff) or len(config.ispBlowDownCoeff):
                raise ValueError("DummyController requires configs without blow-down curves.")
            if not math.isfinite(config.MinOnTime) or config.MinOnTime < 0:
                raise ValueError("MinOnTime must be finite and nonnegative.")

        self.set_enabled_engines(enabled_engines)
        self.command_msgs = {}
        for name, engine in self.engines.items():
            msg = messaging.THRArrayOnTimeCmdMsg()
            self.command_msgs[name] = msg
            engine.subscribe_to_commands(msg)

    @property
    def enabled_engines(self):
        return self._enabled_engines

    def set_enabled_engines(self, names):
        """Replace the selected engines; changes take effect on the next tick."""
        if isinstance(names, str):
            names = [names]
        selected = frozenset(names)
        unknown = selected.difference(self.engines)
        if unknown:
            raise ValueError(f"Unknown engines: {sorted(unknown)}")
        self._enabled_engines = selected

    def set_engine_enabled(self, name, enabled=True):
        """Enable/disable one engine without changing the other selections."""
        if name not in self.engines:
            raise ValueError(f"Unknown engine: {name}")
        selected = set(self._enabled_engines)
        if enabled:
            selected.add(name)
        else:
            selected.discard(name)
        self.set_enabled_engines(selected)

    def Reset(self, CurrentSimNanos):
        self.UpdateState(CurrentSimNanos)

    def UpdateState(self, CurrentSimNanos):
        for name, engine in self.engines.items():
            command = messaging.THRArrayOnTimeCmdMsgPayload()
            # Overlap successive requests so every integration substep sees
            # continuous thrust. Each effector contains one engine: index 0.
            on_time = max(2.0 * self.period_s, engine.config.MinOnTime)
            command.OnTimeRequest = [on_time if name in self._enabled_engines else 0.0]
            self.command_msgs[name].write(command, CurrentSimNanos, self.moduleID)
