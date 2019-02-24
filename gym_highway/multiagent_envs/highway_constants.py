WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED =   (255, 0, 0)
GREY = (210, 210 ,210)
PURPLE = (255, 0, 255)

WIDTH = 1900
HEIGHT = 240
NUM_LANES = 3
LANE_WIDTH = int(HEIGHT/NUM_LANES)
ACTION_RESET_TIME = 0.25 # time till next action
NGSIM_RESET_TIME = 0.1

ppu = 32
car_lane_ratio = 3.7/1.8
CAR_HEIGHT = int((HEIGHT/3.0)/car_lane_ratio)
CAR_WIDTH = int(CAR_HEIGHT*2)

# lane center positions
LANE_1_C = (LANE_WIDTH * 1 - (LANE_WIDTH/2))/ppu
LANE_2_C = (LANE_WIDTH * 2 - (LANE_WIDTH/2))/ppu
LANE_3_C = (LANE_WIDTH * 3 - (LANE_WIDTH/2))/ppu

NEW_LANES = [LANE_1_C, LANE_2_C, LANE_3_C]