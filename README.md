<img width="948" height="910" alt="image" src="https://github.com/user-attachments/assets/31a6730a-7232-4dfc-84f7-1e9d915c6c58" /><img width="948" height="910" alt="image" src="https://github.com/user-attachments/assets/330cf965-a24b-4ac1-bf93-8e7b16281a5e" /># Lunar-Simulation-Environment
A testbed for general lunar landing sequence, including 6-DOF vehicle dynamics, realistic lunar terrain data, and realistic sensor readings. By default, the simulation will place the spacecraft in a lunar descent environment and use an Apollo Powered Descent Guidance Calculation to descend to the lunar surface. Solves the APDG calculation for ToF via an optimization function. Actual control happens through a PD controller. Actual noisy sensor readings from the spacecraft are sent through the Kalman filter in order to get state estimates. 

Lunar Terrain OBJ Download: https://myerauedu-my.sharepoint.com/personal/clayd6_my_erau_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fclayd6%5Fmy%5Ferau%5Fedu%2FDocuments%2FDocuments%2FSERVAL%20LuSE%20Terrain&ga=1

As LuSE is still in development, there is no executable. The simulation executes through a Python IDE, such as Visual Studio Code or PyCharm in runSim.py.

For the development of this code, the PyCharm Code Editor was used. Ensure that you have the lunar terrain downloaded from this source and put it in the Objects folder inside of Environment: SERVAL LuSE Terrain. Name the lunar terrain "SouthPole_Defragged.obj". 

In order to import a custom lander model, although this is not required, upload a .obj with the name "IM1.obj" into the Spacecraft folder. It may take some adjustments to get the spacecraft sizing right, which you can find in Visualization.py in the "Spacecraft Model Scaling + offset" section. 

Upcoming Features:
  Sensor Faults
  Fault Detection, Isolation and Recovery (FDIR)
  Machine Learning based FDIR (ML-FDIR)
  Increase accuracy to approach as close as possible to real lunar descent trajectories and circumstances
