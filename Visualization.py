import os
import numpy as np
from types import SimpleNamespace
from Basilisk.architecture import messaging, sysModel
from Basilisk.utilities import vizSupport
import Spacecraft.Mujoco.FrameTransforms as TerrainFrame
from Basilisk.simulation import vizInterface
from Basilisk.utilities import macros
from Spacecraft.Mujoco.MujocoPhysicsEngine import OBJ_NAME
from Environment.VizardTerrain import prepare_vizard_terrain, prepare_vizard_model, centered_terrain_offset

class _ThrusterVizTelemetry(sysModel.SysModel):
    """Keep hardware ratings constant in visualization while actual thrust varies."""
    def __init__(self, engines):
        super().__init__()
        self.ModelTag = "ThrusterVizTelemetry"
        self.entries = []
        self.proxies = []
        for engine in engines.values():
            output = messaging.THROutputMsg()
            self.entries.append((engine, float(engine.config.MaxThrust), output,
                                 engine.thruster.thrusterOutMsgs[0].addSubscriber()))
            self.proxies.append(SimpleNamespace(ModelTag=engine.name, thrusterOutMsgs=[output]))

    def Reset(self, now):
        self.UpdateState(now)

    def UpdateState(self, now):
        for engine, rating, output, source in self.entries:
            payload = source() if source.isWritten() else messaging.THROutputMsgPayload()
            payload.maxThrust = rating
            if not source.isWritten():
                payload.thrusterLocation = np.asarray(engine.config.thrLoc_B).reshape(3).tolist()
                payload.thrusterDirection = np.asarray(engine.config.thrDir_B).reshape(3).tolist()
            output.write(payload, now, self.moduleID)




def initialize_vizard(sim, sc, task_name="record", save_file=None, terrain_grid_cells=180, model_grid_cells=32):

    if vizSupport.vizFound:
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "Spacecraft", "IM1.obj"))
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Vizard spacecraft model not found: {model_path}")
        model_path = prepare_vizard_model(
            model_path, os.path.join(os.path.dirname(__file__), 'work', 'vizard_models'),
            grid_cells=model_grid_cells).path

        # =========================================================
        # IMX LIGHT
        # =========================================================
        imx_light = vizInterface.Light()

        # Basilisk's LightVector stores raw Light* pointers, not owning copies.
        # Keep the Python owner alive until the simulation is released; a local
        # light can otherwise be freed before Vizard's first native update.
        if not hasattr(sim, "_vizard_lights"):
            sim._vizard_lights = []
        sim._vizard_lights.append(imx_light)

        imx_light.label = "IMX Light"

        # Position relative to IMX spacecraft body frame
        imx_light.position = [0.0, 0.0, -1.0]

        # Point in -Z body direction
        imx_light.normalVector = [0.0, 0.0, -1.0]

        # 90 degree field of view
        imx_light.fieldOfView = 90.0 * macros.D2R

        # Maximum light range [m]
        imx_light.range = 500.0

        # Light intensity
        imx_light.intensity = 1.0

        # Don't show light position marker
        imx_light.showLightMarker = -1

        # Don't show lens flare
        imx_light.showLensFlare = -1

        # =========================================================
        # CREATE VIZARD
        # =========================================================
        telemetry = _ThrusterVizTelemetry(sc.engines)
        sim._vizard_thruster_telemetry = telemetry
        sim.AddModelToTask(task_name, telemetry, ModelPriority=1)
        terrain = getattr(sc, 'terrain', None)
        has_terrain = terrain is not None
        viz = vizSupport.enableUnityVisualization(
            sim,
            task_name,
            [sc.lander, terrain] if has_terrain else [sc.lander],
            saveFile=save_file if save_file is not None else __file__,
            thrEffectorList=[telemetry.proxies, None] if has_terrain else [telemetry.proxies],
            thrColors=[[
                [80, 160, 255, 255]  # Main engine: blue
                if engine.name == "MainEngine"
                else [255, 255, 255, 255]  # RCS: white
                for engine in sc.engines.values()
            ]] + ([None] if has_terrain else []) if sc.engines else None,

            # First list = sc.lander lights
            # Second list = sc.terrain lights
            lightList=[[imx_light], []] if has_terrain else [[imx_light]]
        )

        # vizSupport normally discovers only the plant's gravitational bodies.
        # A Moon-only export crashes Vizard's Moon orbit initialization because
        # its Earth parent index is -1. These extra bodies affect display only.
        body_info = list(viz.gravBodyInformation)
        body_inputs = list(viz.spiceInMsgs)
        names = {body.bodyName.lower() for body in body_info}
        for body in getattr(sc, "visualization_bodies", []):
            name = body.displayName or body.planetName
            if name.lower() in names:
                continue
            info = vizInterface.GravBodyInfo()
            info.bodyName = name
            info.mu = body.mu
            info.radEquator = body.radEquator
            info.radiusRatio = body.radiusRatio
            info.modelDictionaryKey = body.modelDictionaryKey
            body_info.append(info)
            body_inputs.append(body.planetBodyInMsg)
            names.add(name.lower())
        viz.gravBodyInformation = vizInterface.GravBodyInfoVector(body_info)
        viz.spiceInMsgs = messaging.SpicePlanetStateMsgInMsgsVector(body_inputs)
        if hasattr(sc, "gravity_factory"):
            viz.epochInMsg.subscribeTo(sc.gravity_factory.epochMsg)

        viz.settings.showSpacecraftLabels = 1
        # Load this bounded 5 Hz recording entirely. The default 10 MB buffer
        # rolls over near 13:36 in this scenario; avoid the player's refill path.
        viz.settings.messageBufferSize = -1
        viz.settings.showSpacecraftAsSprites = -1
        viz.settings.ambient = 1.5
        viz.settings.spacecraftShadowBrightness = 0.07

        vizSupport.setActuatorGuiSetting(viz)

        # Spacecraft Model Scaling + offset
        vizSupport.createCustomModel(
            viz,
            model_path,
            simBodiesToModify=[sc.lander.ModelTag],
            offset=[0,0,-1.2],
            # The replacement OBJ is already metre-scale (about 1.7 m tall).
            scale=[1.15, 1.15, 1.15],
            color=(255, 0, 0, 255)
        )

        # LUNAR TERRAIN MODEL
        if has_terrain:
            terrain_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "Environment",
                    "Objects",
                    f"{OBJ_NAME}"
                )
            )

            terrain_map = getattr(sc, "terrain_map", None)
            if terrain_map is not None:
                terrain_path = terrain_map.export_vizard_obj(os.path.join(
                    os.path.dirname(__file__), "work", "terrain_collision_aligned.obj"), versioned=True)

            display = getattr(sc, 'vizard_terrain', None)
            if display is None or terrain_map is not None:
                display = prepare_vizard_terrain(
                    terrain_path, os.path.join(os.path.dirname(__file__), 'work', 'vizard_terrain'),
                    grid_cells=terrain_grid_cells, surface_sign=-1 if terrain_map is not None else 1)
                sc.vizard_terrain = display
            terrain_path = display.path
            # Collision exports already contain the centered world positions.
            # Raw OBJ previews require the shared source-corner correction.
            terrain_offset = ([0., 0., 0.] if terrain_map is not None else
                              centered_terrain_offset(display, TerrainFrame.TERRAIN_ROTATION_VIZARD,
                                                      TerrainFrame.TERRAIN_SCALE))

            vizSupport.createCustomModel(
                viz,
                terrain_path,
                simBodiesToModify=[terrain.ModelTag],
                offset=terrain_offset,
                rotation=[0.,0.,0.] if terrain_map is not None else TerrainFrame.TERRAIN_ROTATION_VIZARD,
                scale=[1.,1.,1.] if terrain_map is not None else TerrainFrame.TERRAIN_SCALE
            )

        # =========================================================
        # CAMERA SETTINGS
        # =========================================================
        viz.settings.forceStartAtSpacecraftLocalView = 1
        viz.settings.mainCameraTarget = sc.lander.ModelTag

        if getattr(sc, 'model_check_only', False):
            # Body +Z is lunar outward/up in the preview; -Z looks at the
            # terrain, with the camera below the lander to avoid CAD occlusion.
            vizSupport.createStandardCamera(
                viz, spacecraftName=sc.lander.ModelTag, setMode=1,
                pointingVector_B=[0., 0., -1.], position_B=[15., 0., -3.],
                fieldOfView=60. * macros.D2R, displayName='Terrain below IMX')

        viz.settings.scViewToPlanetViewBoundaryMultiplier = 10
        viz.settings.planetViewToHelioViewBoundaryMultiplier = 10
        return viz
