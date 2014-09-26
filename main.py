# OpenNero will execute ModMain when this mod is loaded
import OpenNero
import common

import TowerofHanoi.main
import TowerofHanoi.constants
from TowerofHanoi.module import getMod, TowerEnvironment
import Hw2.agent

def start_my_planner_2_disk():
    """ start the tower demo """
    getMod().num_disks = 2
    OpenNero.disable_ai()
    getMod().stop_agent()
    env = TowerEnvironment()
    env.initialize_blocks()
    getMod().set_environment(env)
    getMod().agent_id = common.addObject("data/shapes/character/MyPlanningRobot2.xml", OpenNero.Vector3f(TowerofHanoi.constants.GRID_DX, TowerofHanoi.constants.GRID_DY, 2), type=TowerofHanoi.constants.AGENT_MASK, scale=OpenNero.Vector3f(3,3,3))
    OpenNero.enable_ai()

def start_my_planner_3_disk():
    """ start the tower demo """
    getMod().num_disks = 3
    OpenNero.disable_ai()
    getMod().stop_agent()
    env = TowerEnvironment()
    env.initialize_blocks()
    getMod().set_environment(env)
    getMod().agent_id = common.addObject("data/shapes/character/MyPlanningRobot3.xml", OpenNero.Vector3f(TowerofHanoi.constants.GRID_DX, TowerofHanoi.constants.GRID_DY, 2), type=TowerofHanoi.constants.AGENT_MASK, scale=OpenNero.Vector3f(3,3,3))
    OpenNero.enable_ai()
        
def ModMain():
    agent_desc = ('My Strips Planner - 2 Disks', start_my_planner_2_disk, "")
    getMod().AGENTS.append(agent_desc)
    agent_desc = ('My Strips Planner - 3 Disks', start_my_planner_3_disk, "")
    getMod().AGENTS.append(agent_desc)
    TowerofHanoi.main.ModMain()

