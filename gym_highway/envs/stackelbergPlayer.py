import os
import pygame
import copy
from math import tan, radians, degrees, copysign, exp
from pygame.math import Vector2
from enum import Enum

# TODO: probably set these values during init
VISIBLE_DIST = 1900.0 # pixels
LANE_DIFF = 120.0 # 120 pixels between centers of lanes
ACTION_HORIZON = 0.25
COMFORT_LVL = 0.0
NUM_PLAYERS = 3
NUM_LANES = 3

class Action(Enum):
    LEFT = 0
    RIGHT = 1
    MAINTAIN = 2
    ACCELERATE = 3
    DECELERATE = 4

class StackelbergPlayer():
    def __init__(self, car_width):
        self.car_width = car_width/64
        self.players = [set() for x in range(NUM_PLAYERS)]
        self.playerSets = {}

    def selectAction(self, leader, all_obstacles):
        selected_action = self.getActionUtilSet(leader, all_obstacles)[0][0]
        return selected_action

    def selectStackelbergAction(self, leader, all_obstacles, reference_car):

        # Step 1: get the players involved in this game
        # --------------------------------------------------------------------------------------------
        s_players = self.playerSets[leader]

        # keep track of the ids for each agent so they can be cloned
        p_ids = []
        for obstacle in all_obstacles:
            if obstacle in s_players:
                p_ids.append(obstacle.id)

        # actions and player count
        all_actions = list(Action)
        p_count = len(s_players)

        # Step 2: create copies of all the obstacles so we can project their actions
        # --------------------------------------------------------------------------------------------
        # state buffer to revert actions
        state_buffer = [[] for x in range(min(2,p_count))]
        # make a copy of all the current obstacles
        state_buffer[0] = self.copyAllObjects(all_obstacles)
        
        # save references to the players involved in the current Stackelberg game
        mut_agents = [obstacle_copy for obstacle_copy in state_buffer[0] if obstacle_copy.id in p_ids]
        # make sure the players are sorted
        mut_agents = self.sortByPosition(mut_agents)

        # Step 3: initialize all the default utilities and actions
        # --------------------------------------------------------------------------------------------
        # p1_best, p2_best = float("-inf"), float("-inf")
        p1_best, p2_best = 0.0, 0.0
        p1_best_action, p2_best_action, p3_best_action = Action.MAINTAIN, Action.MAINTAIN, Action.MAINTAIN

        p1_action_list = self.getActionSubset(all_actions, mut_agents[0])
        if p_count > 1:
            p2_action_list = self.getActionSubset(all_actions, mut_agents[1])

        # Step 4: get the best action for the leader using the Stackelberg game:
        # --------------------------------------------------------------------------------------------
        for p1_action in p1_action_list:

            # 1a. select and execute an action for player[0]
            self.executeAction(p1_action, mut_agents[0], state_buffer[0])
            # update state_buffer so that it does not contain the 2 other vehicles for projection
            no_update_players = []
            if p_count > 1:
                no_update_players = mut_agents[1:]
            for player in state_buffer[0]:
                if player not in no_update_players:
                    player.update(ACTION_HORIZON, reference_car)
            # state_buffer[0].update(ACTION_HORIZON, reference_car)

            # if player 2 exists, consider its response to player 1's action
            if p_count > 1:
                
                # make a copy of the state induced by player 1's action
                state_buffer[1] = self.copyAllObjects(state_buffer[0])
                # since new copies are generated, we need to update the Stackelberg player copies
                mut_agents = [obstacle_copy for obstacle_copy in state_buffer[1] if obstacle_copy.id in p_ids]
                mut_agents = self.sortByPosition(mut_agents)

                # 1b. observe how followers react to selected action
                for p2_action in p2_action_list:
                    
                    # 2a. select and execute an action for player[1]
                    self.executeAction(p2_action, mut_agents[1], state_buffer[1])
                    mut_agents[1].update(ACTION_HORIZON, reference_car)
                    # state_buffer[1].update(ACTION_HORIZON, reference_car)

                    # if player 3 exists, consider its response to player 1 and 2's action
                    p3_action = None
                    if p_count > 2:
                        # 3a. select and execute an action which maximizes player[2] utility
                        p3_action = self.selectAction(mut_agents[2], state_buffer[1])
                        self.executeAction(p3_action, mut_agents[2], state_buffer[1])
                        mut_agents[2].update(ACTION_HORIZON, reference_car)
                        # state_buffer[1].update(ACTION_HORIZON, reference_car)

                    # 2b. calculate utility value for player 2 in the current state
                    p2_utility = self.positiveUtility(mut_agents[1], mut_agents[1].lane_id, mut_agents[1].velocity.x, state_buffer[1])
                    p2_utility += self.negativeUtility(mut_agents[1], mut_agents[1].lane_id, mut_agents[1].velocity.x, state_buffer[1])

                    # 2c. select action which results in the best utility value
                    if p2_utility > p2_best:
                        p2_best = p2_utility
                        # update player 2 and 3 best actions
                        p2_best_action = p2_action
                        p3_best_action = p3_action

                    # revert state back to the one generated by player 1's action
                    state_buffer[1] = self.copyAllObjects(state_buffer[0])
                    mut_agents = [obstacle_copy for obstacle_copy in state_buffer[1] if obstacle_copy.id in p_ids]
                    mut_agents = self.sortByPosition(mut_agents)

                # execute the best actions for player 2 and 3
                mut_agents = [obstacle_copy for obstacle_copy in state_buffer[0] if obstacle_copy.id in p_ids]
                mut_agents = self.sortByPosition(mut_agents)
                self.executeAction(p2_best_action, mut_agents[1], state_buffer[0])
                # DEBUG: FOUND BUG
                mut_agents[1].update(ACTION_HORIZON, reference_car)
                # state_buffer[0].update(ACTION_HORIZON, reference_car)

                if p_count > 2:
                    self.executeAction(p3_best_action, mut_agents[2], state_buffer[0])
                    # DEBUG: FOUND BUG
                    mut_agents[2].update(ACTION_HORIZON, reference_car)
                    # state_buffer[0].update(ACTION_HORIZON, reference_car)

            # TODO remove after testing
            # print(mut_agents[0].velocity.x)

            # 1c. calculate utility value for final state
            p1_utility = self.positiveUtility(mut_agents[0], mut_agents[0].lane_id, mut_agents[0].velocity.x, state_buffer[0])
            p1_utility += self.negativeUtility(mut_agents[0], mut_agents[0].lane_id, mut_agents[0].velocity.x, state_buffer[0])

            # 1d. select the action which results in the best utility value
            if p1_utility > p1_best:
                p1_best = p1_utility
                p1_best_action = p1_action

            # reset the state for agents 1, 2 and 3
            # self.resetState(mut_agents, s_players, all_obstacles_copy, p_range)
            state_buffer[0] = self.copyAllObjects(all_obstacles)
            mut_agents = [obstacle_copy for obstacle_copy in state_buffer[0] if obstacle_copy.id in p_ids]
            mut_agents = self.sortByPosition(mut_agents)

        # print(p1_best_action.name)
        return p1_best_action

    def resetState(self, mut_agents, s_players, all_obstacles_copy, resetList):
        # for obstacle_copy in mut_agents:
        #     # if obstacle_copy.id in p_ids:

        for i in resetList:
            all_obstacles_copy.remove(mut_agents[i])
            mut_agents[i] = s_players[i].simCopy()
            all_obstacles_copy.append(mut_agents[i])

        return

    def copyAllObjects(self, all_objects):
        copied_objects = pygame.sprite.Group()
        for obj in all_objects:
            copied_objects.add(obj.simCopy())
        return copied_objects

    def getActionUtilSet(self, leader, all_obstacles):
        current_lane = leader.lane_id
        all_actions = list(Action)

        # print(all_obstacles)

        # if in the left lane remove left action, same with right lane
        if current_lane == 1:
            del all_actions[Action.LEFT.value]
        elif current_lane == 3:
            del all_actions[Action.RIGHT.value]

        best_utility = 0.0
        selected_action = Action.MAINTAIN
        # selected_action = None
        action_util_dict = {}
        for action in all_actions:
            # update the intended lane
            intended_lane = current_lane
            if action == Action.LEFT:
                intended_lane -= 1
            elif action == Action.RIGHT:
                intended_lane += 1

            # update intended velocity
            intended_velocity = self.updatedVelocity(leader, action)
            # compute utility for the current action
            current_utility = self.positiveUtility(leader, intended_lane, intended_velocity, all_obstacles)
            current_utility += self.negativeUtility(leader, intended_lane, intended_velocity, all_obstacles)
            if current_utility > best_utility:
                best_utility = current_utility
                selected_action = action

            # save action and corresponding utility
            action_util_dict.update({action:current_utility})

        action_util_sorted = sorted(action_util_dict.items(), key=lambda kv: kv[1], reverse=True)
        # print(action_util_sorted[0][1])
        
        return action_util_sorted
        # return selected_action

    def getActionSubset(self, actions, ego):
        # if in the left lane remove left action, same with right lane
        actions_subset = actions.copy()
        if ego.lane_id == 1:
            actions_subset.remove(Action.LEFT)
        elif ego.lane_id == 3:
            actions_subset.remove(Action.RIGHT)
        return actions_subset

    # select leaders to make a decision at this instance, and followers for next iteration
    def pickLeadersAndFollowers(self, all_agents, all_obstacles):
        # 1. rank all players based on position on road
        sorted_agents = self.sortByPosition(all_agents)
        
        # 2. select top agent as leader and add to leader/follower list
        while sorted_agents:
            leader = sorted_agents.pop(0)

            # get the followers for this leader
            all_players = self.pickPlayers(leader, all_agents, all_obstacles)

            # 3. add the leaders and followers to the players list of sets
            for idx, agent in enumerate(all_players):
                self.players[idx].add(agent)

            # 4. remove all these from the original sorted list
            sorted_agents = [agent for agent in sorted_agents if agent not in all_players]        
        
        leader_list = self.players[0]

        # save player sets for each leader
        # TODO: testing making everyone a leader
        for leader in leader_list:
        # for leader in sorted_agents:
            # get the followers for this leader
            all_players = self.pickPlayers(leader, all_agents, all_obstacles)
            self.playerSets[leader] = all_players

        # update players for next turn
        # TODO: commented this out for testing, undo if new idea doesnt work
        self.updatePlayersList()

        # return the leaders
        # TODO: testing using all agents as leaders
        return leader_list
        # return sorted_agents

    # remove current leaders and make first followers the new leaders
    # TODO: might want to make this into a queue datastructure
    def updatePlayersList(self):
        # remove current leaders
        del self.players[0]
        # add empty set for new followers
        self.players.append(set())
        return

    # Get at most 3 vehicles that are close to the back of the ego vehicle
    def pickPlayers(self, ego, all_agents, all_obstacles):
        players = [ego]

        # by default it is the middle lane
        adversary_lane = 2
        
        # need to update adversary lane if leader is already in the middle lane
        if ego.lane_id == 2:
            action_util_set = self.getActionUtilSet(ego, all_obstacles)
            for action_tuple in action_util_set:
                if action_tuple[0] == Action.LEFT:
                    adversary_lane -= 1
                    break
                elif action_tuple[0] == Action.RIGHT:
                    adversary_lane += 1
                    break

        # the two other players considered
        back_agent, side_agent = None, None
        for agent in all_obstacles:
            if agent != ego:
                # agent in the same lane
                if agent.lane_id == ego.lane_id:
                    # agent has to be behind the ego vehicle
                    if agent.position.x < ego.position.x:
                        if not back_agent:
                            back_agent = agent
                        # agent closest to the back
                        elif agent.position.x > back_agent.position.x:
                            back_agent = agent
                # agent is in adjacent lane
                elif agent.lane_id == adversary_lane:
                    # agent has to be behind the ego vehicle
                    if agent.position.x < ego.position.x:
                        if not side_agent:
                            side_agent = agent
                        # agent closest the to back side
                        elif agent.position.x > side_agent.position.x:
                            side_agent = agent
        if back_agent in all_agents: players.append(back_agent)
        if side_agent in all_agents: players.append(side_agent)

        # return players sorted by their longitudinal position in decending order
        return self.sortByPosition(players)

    def updatedVelocity(self, ego, action):
        intended_velocity = ego.velocity.x

        # increase or decrease velocity (vf = vi + accel*time), assume accel of 1
        if action == Action.ACCELERATE:
            intended_velocity += 1*ACTION_HORIZON
        elif action == Action.DECELERATE:
            intended_velocity -= 1*ACTION_HORIZON
        return max(0.0, min(intended_velocity, ego.max_velocity))

    # sort vehicles by longitudinal position (decreasing order)
    def sortByPosition(self, all_agents):
        sorted_agents = sorted(all_agents, key=lambda x: x.position.x, reverse=True)
        return sorted_agents

    # TODO: start with the simple positive utility
    def positiveUtility(self, ego, intended_lane, intended_velocity, all_obstacles):
        # max stopping distance
        ideal_distance = self.stoppingDist(ego, ego.max_velocity)

        for obstacle in all_obstacles:
            if obstacle == ego:
                continue
            if obstacle.lane_id == intended_lane:
                dx = obstacle.position.x - ego.position.x
                # dx = (obstacle.position.x - (obstacle.rect[2]/64)) - (ego.position.x + (ego.rect[2]/64)) - COMFORT_LVL

                # only consider vehicles ahead of ego vehicle
                if dx >= 0:
                    # calculate actual difference between cars
                    dx = abs(obstacle.position.x - ego.position.x) - (ego.rect[2]/32) - self.car_width

                    # TODO: try adding difference in lateral positions
                    # dy = 0.0
                    dy = abs(obstacle.position.y - ego.position.y)
                    dy = LANE_DIFF/32 if dx > ego.rect[2]/32 else dy

                    stopping_dist = self.stoppingDist(ego, intended_velocity)
                    # tmp_val = stopping_dist + min(dx - stopping_dist, 0)*10
                    tmp_val = stopping_dist - exp(0.5*abs(min(dx - stopping_dist, 0))) - exp(-1*dy)

                    ideal_distance = min(tmp_val, ideal_distance)
        return ideal_distance

    # compute stopping distance for ego vehicle
    def stoppingDist(self, ego, intended_velocity):
        return 0.5*(intended_velocity ** 2)/ego.max_acceleration

    def negativeUtility(self, ego, intended_lane, intended_velocity, all_obstacles):
        neg_utility = None
        for obstacle in all_obstacles:
            if obstacle == ego:
                continue
            if obstacle.lane_id == intended_lane:
                dx = obstacle.position.x - ego.position.x
                # dx = (obstacle.position.x + (obstacle.rect[2]/64)) - (ego.position.x - (ego.rect[2]/64)) + COMFORT_LVL

                # only consider vehicles behind of ego vehicle
                if dx <= 0:
                    dx = abs(obstacle.position.x - ego.position.x) - (ego.rect[2]/32) - self.car_width

                    dv = obstacle.velocity.x - intended_velocity    
                    # dv = obstacle.velocity.x - ego.velocity.x

                    time_lane_change = self.timeToChangeLane(ego, intended_velocity)
                    dist_lane_change = intended_velocity * time_lane_change
                    # dist_lane_change = ego.velocity.x * time_lane_change

                    # TODO: try adding difference in lateral positions
                    # dy = 0.0
                    dy = abs(obstacle.position.y - ego.position.y)
                    dy = LANE_DIFF/32 if dx > ego.rect[2]/32 else dy

                    # Negative utility formula
                    if not neg_utility:
                        neg_utility = dx - dv*time_lane_change - dist_lane_change - exp(-1*dy)
                    else:
                        neg_utility = min(dx - dv*time_lane_change - dist_lane_change - exp(-1*dy), neg_utility) 
                    # neg_utility = abs(dx) - dv*time_lane_change - dist_lane_change

        # set neg_utility to 0.0 if it was not assigned above
        neg_utility = neg_utility if neg_utility else 0.0
        return neg_utility

    # Calculate lateral velocity assuming max steering for vehicle to get time to change lane
    def timeToChangeLane(self, ego, intended_velocity):
        turning_radius = ego.length / tan(radians(ego.max_steering))
        # prevent intended velocity from being zero
        angular_velocity = max(intended_velocity, 1.0) / turning_radius

        # assuming center of lanes, we know the distance is 120 pixels
        lane_change_time = LANE_DIFF/degrees(angular_velocity)
        return lane_change_time

    # TODO: really need to reuse these from Game(), need to understand inheritance in python
    # execute the given action for the specified leader
    def executeAction(self, selected_action, leader, all_obstacles):
        if (selected_action == Action.ACCELERATE) and not leader.do_accelerate:
            self.accelerate(leader)
        elif (selected_action == Action.DECELERATE) and not leader.do_decelerate:
            self.decelerate(leader)
        elif (selected_action == Action.MAINTAIN) and not leader.do_maintain:
            self.maintain(leader, all_obstacles)

        leader.acceleration = max(-leader.max_acceleration, min(leader.acceleration, leader.max_acceleration))

        if (selected_action == Action.RIGHT) and not leader.right_mode:
            self.turn_right(leader)
        elif (selected_action == Action.LEFT) and not leader.left_mode:
            self.turn_left(leader)

        leader.steering = max(-leader.max_steering, min(leader.steering, leader.max_steering))
        
        return

    # these booleans are required to ensure the action is executed over a period of time
    def accelerate(self, car):
        car.do_accelerate = True
        car.do_decelerate = False
        car.do_maintain = False

    def maintain(self, car, all_obstacles):
        # only execute this function when required
        if car.do_maintain:
            return

        forward_obstacle = None
        for obstacle in all_obstacles:
            if obstacle == car:
                continue
            # obstacle in the same lane
            if obstacle.lane_id == car.lane_id:
                # obstacle has to be ahead the ego vehicle
                if obstacle.position.x > car.position.x:
                    if not forward_obstacle:
                        forward_obstacle = obstacle
                    # obstacle closest to the front
                    elif obstacle.position.x < forward_obstacle.position.x:
                        forward_obstacle = obstacle
        
        obstacle_velx = forward_obstacle.velocity.x if forward_obstacle else car.velocity.x
        car.setCruiseVel(obstacle_velx)
        car.do_maintain = True
        car.do_accelerate = False
        car.do_decelerate = False

    def decelerate(self, car):
        car.do_decelerate = True
        car.do_accelerate = False
        car.do_maintain = False

    def turn_right(self, car):
        car.lane_id = min(car.lane_id + 1, NUM_LANES)
        car.right_mode = True
        car.left_mode = False

    def turn_left(self, car):
        car.lane_id = max(car.lane_id - 1, 1)
        car.left_mode = True
        car.right_mode = False
