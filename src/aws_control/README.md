# ITF-AIRP 4WS transportation vehicle control

link to setup python environment
https://docs.omniverse.nvidia.com/isaacsim/latest/install_python.html?highlight=python%20sh 

- prerequisites
    ```
    conda env create -f environment.yml (/workspace/4WS_control/ISAAC_SIM/environment.yml)
    export ISAACSIM_SA=<poath to isaac_sim>/setup_conda_env.sh
    source $ISAACSIM_SA # in every new terminal
    
    ### upper commonds not working, use the following methods to run isaac related scripts
    .ISAAC_SIM/python.sh path/to/script.py
    .ISAAC_SIM/python.sh -m pip install name_of_package_here

    ```

- test isaac_sim AGV environment
    ```
    python AGV_env.py
    ```

- AGV environment ros interface
    ```
    python ros_interface.py
    rviz -d rviz.rviz
    ```
    use Omniverse -> Issac-Sim-2022.2.1 unzip and open AGV.usd
    
    demo ros messgae:
    ```
    rostopic pub -r 1 /robot_control_command geometry_msgs/Twist "linear:
    x: 0.0
    y: 0.0
    z: 0.0
    angular:
    x: 0.0
    y: 0.0
    z: -0.5" 

    ```

- test mpc follower
    ```
    # To record path in a bag. The following commond can record /Imu_Sensor in 10Hz frame rate
    python plan_utils/routing.py --record
    # Useage
    roscore 
    conda activate AIRP
    rviz -d plan_config.rviz 
    python plan_utils/ros_interface.py 
    python plan_utils/routing.py 
    python path_follower.py

    ```

- Utils

    Sometimes the Omniverse have some rubbish thread in RAM, use
    ```
    sudo kill $(ps aux | grep "omni" | grep -v grep | awk '{print $2}')
    ``` 
    to kill them all and restart Omniverse.


- hloc
    ``` 
    sudo docker run -it --name itf-hloc --gpus '0'  --network host -e DISPLAY --privileged -v       /home/ubuntu/workspace/itf:/workspace/ iidcramlab/itf-airport:hloc
    ```


# TODO
- [ ] Ros launch
- [ ] Step by step simulation


# others
- 修改传感器帧率 http://zhaoxuhui.top/blog/2023/02/12/omniverse-and-isaac-sim-note8-camera-frequency-setting.html