import os
from Basilisk.utilities import vizSupport
import Spacecraft.Mujoco.FrameTransforms as TerrainFrame
from Basilisk.simulation import vizInterface
from Basilisk.utilities import macros
from Spacecraft.Mujoco.MujocoPhysicsEngine import OBJ_NAME

def initialize_vizard(sim, sc):

    if vizSupport.vizFound:

        # =========================================================
        # IMX LIGHT
        # =========================================================
        imx_light = vizInterface.Light()

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
        viz = vizSupport.enableUnityVisualization(
            sim,
            "record",
            [sc.lander, sc.terrain],
            saveFile=__file__,

            # First list = sc.lander lights
            # Second list = sc.terrain lights
            lightList=[[imx_light], []]
        )

        viz.settings.showSpacecraftLabels = 1
        viz.settings.showSpacecraftAsSprites = -1
        viz.settings.ambient = 1.5
        viz.settings.spacecraftShadowBrightness = 0.07

        vizSupport.setActuatorGuiSetting(viz)

        # =========================================================
        # REAL SPACECRAFT MODEL
        # =========================================================
        model_path = os.path.abspath(
            os.path.join("Spacecraft", "IM1.obj")
        )

        vizSupport.createCustomModel(
            viz,
            model_path,
            simBodiesToModify=[sc.lander.ModelTag],
            offset=[0.73, -0.73, -1.05],
            scale=[0.0004, 0.0004, 0.0004],
            color=(1, 0, 0, 1)
        )

        # =========================================================
        # LUNAR TERRAIN MODEL
        # =========================================================
        terrain_path = os.path.abspath(
            os.path.join(
                "Environment",
                "Objects",
                f"{OBJ_NAME}"
            )
        )

        vizSupport.createCustomModel(
            viz,
            terrain_path,
            simBodiesToModify=[sc.terrain.ModelTag],
            offset=[0.0, 0.0, 0.0],
            rotation=TerrainFrame.TERRAIN_ROTATION_VIZARD,
            scale=TerrainFrame.TERRAIN_SCALE
        )

        # =========================================================
        # CAMERA SETTINGS
        # =========================================================
        viz.settings.forceStartAtSpacecraftLocalView = 1

        viz.settings.scViewToPlanetViewBoundaryMultiplier = 10
        viz.settings.planetViewToHelioViewBoundaryMultiplier = 10