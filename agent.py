from OpenNero import *
from common import *

import TowerofHanoi 
from TowerofHanoi.constants import *
from TowerofHanoi.environment import TowerEnvironment
from TowerofHanoi.constants import *
from copy import copy

ACTIONS_BEGIN = [5]
ACTIONS_CELEBERATE = [0,0,0,5,5,1]

class MyPlanningAgent2Disk(AgentBrain):#2-Disk Strips Planner
    """
    An agent that uses a STRIPS planner to solve the Tower of Hanoi problem for 2 disks
    """
    def __init__(self):
        AgentBrain.__init__(self) # have to make this call

    def initialize(self,init_info):
        """
        Create the agent.
        init_info -- AgentInitInfo that describes the observation space (.sensors),
                     the action space (.actions) and the reward space (.rewards)
        """
        self.action_info = init_info.actions
        return True

    def generate_action_list(self):
        import subprocess
        # solve for show (user can click through)
        subproc = subprocess.Popen(['python', 'Hw2/mystrips.py', 'TowerofHanoi/towers2_strips.txt'], stdout=subprocess.PIPE)
        
        plan = ''
        while True:
            try:
                out = subproc.stdout.read(1)
            except:
                break
            if out == '':
                break
            else:
                plan += out
        print plan

        hl_actions = [] # high level action
        for line in plan.split('\n'):
            words = line.strip().split()
            if len(words) == 3:
                (what, frm, to) = words
                hl_actions.append((what, frm, to))
        
        from TowerofHanoi.towers import Towers2 as towers
        
        action_list = []
        action_list.extend(ACTIONS_BEGIN)
        state = copy(towers.INIT)
        at = towers.Pole1
        for (what, frm, to) in hl_actions:
            frm_pole = towers.get_pole(state, frm)
            to_pole = towers.get_pole(state, to)
            print what, frm, to, at, frm_pole, to_pole
            if at != frm_pole:
                action_list += towers.MOVES[(at, frm_pole)]
            action_list += towers.CARRY_MOVES[(frm_pole, to_pole)]
            towers.Move(state, what, frm, to)
            at = to_pole

        action_list.extend(ACTIONS_CELEBERATE)
        return action_list

    def start(self, time, observations):
        """
        return first action given the first observations
        """
        self.action_list = self.generate_action_list()
        if len(self.action_list) > 0:
            return self.action_list.pop(0)
        else:
            return 0

    def act(self, time, observations, reward):
        """
        return an action given the reward for the previous
        action and the new observations
        """
        if len(self.action_list) > 0:
            return self.action_list.pop(0)
        else:
            return 1

    def end(self, time, reward):
        """
        receive the reward for the last observation
        """
        return True

    def reset(self):
        #self.action_list = self.generate_action_list()
        return True

    def destroy(self):
        """
        called when the agent is done
        """
        return True

class MyPlanningAgent3Disk(AgentBrain):#3-Disk Strips Planner 
    """
    An agent that uses a STRIPS planner to solve the Tower of Hanoi problem for 3 disks
    """
    def __init__(self):
        AgentBrain.__init__(self) # have to make this call

    def initialize(self,init_info):
        """
        Create the agent.
        init_info -- AgentInitInfo that describes the observation space (.sensors),
                     the action space (.actions) and the reward space (.rewards)
        """
        self.action_info = init_info.actions
        return True

    def generate_action_list(self):
        import subprocess
        # solve for show (user can click through)
        subproc = subprocess.Popen(['python', 'Hw2/mystrips.py', 'TowerofHanoi/towers3_strips.txt'], stdout=subprocess.PIPE)
        plan = ''
        while True:
            try:
                out = subproc.stdout.read(1)
            except:
                break
            if out == '':
                break
            else:
                plan += out
        print plan
        hl_actions = [] # high level action
        for line in plan.split('\n'):
            words = line.strip().split()
            if len(words) == 3:
                (what, frm, to) = words
                hl_actions.append((what, frm, to))
        
        from TowerofHanoi.towers import Towers3 as towers
        
        action_list = []
        action_list.extend(ACTIONS_BEGIN)
        state = copy(towers.INIT)
        at = towers.Pole1
        for (what, frm, to) in hl_actions:
            frm_pole = towers.get_pole(state, frm)
            to_pole = towers.get_pole(state, to)
            print what, frm, to, at, frm_pole, to_pole
            if at != frm_pole:
                action_list += towers.MOVES[(at, frm_pole)]
            action_list += towers.CARRY_MOVES[(frm_pole, to_pole)]
            towers.Move(state, what, frm, to)
            at = to_pole

        action_list.extend(ACTIONS_CELEBERATE)
        return action_list

    def start(self, time, observations):
        """
        return first action given the first observations
        """
        self.action_list = self.generate_action_list()
        if len(self.action_list) > 0:
            return self.action_list.pop(0)
        else:
            return 0

    def act(self, time, observations, reward):
        """
        return an action given the reward for the previous
        action and the new observations
        """
        if len(self.action_list) > 0:
            return self.action_list.pop(0)
        else:
            return 1

    def end(self, time, reward):
        """
        receive the reward for the last observation
        """
        return True

    def reset(self):
        #self.action_list = self.generate_action_list()
        return True

    def destroy(self):
        """
        called when the agent is done
        """
        return True

