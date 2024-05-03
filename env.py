import gymnasium as gym
from gymnasium import spaces
import pygame
from ddave.utils import *
from ddave.helper import *
import numpy as np
import configparser
from sklearn.preprocessing import LabelEncoder


# Load config file
config = configparser.ConfigParser()
config.read('game.cfg')

RESCALE_FACTOR = int(config['GAME']['RESCALE_FACTOR'])
END_LEVEL_SCORE = int(config['GAME']['END_LEVEL_SCORE'])


class DangerousDaveEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode="human",env_rep_type='image',random_respawn=False,policy='CNN'):
        self.render_mode = render_mode
        self.env_rep_type = env_rep_type
        self.policy = policy
        # Initialize pygame
        pygame.init()
        self.game_screen = Screen(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.random_respawn = random_respawn

        # Initialize tiles
        self.tileset, self.ui_tileset = load_game_tiles()

        # Define action space (movement keys)
        self.action_space = spaces.Discrete(4)  # Up, Left, Right, Down

        self.movement_keys = [pygame.K_UP, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_DOWN]
        self.inv_keys = [pygame.K_LCTRL, pygame.K_RCTRL, pygame.K_LALT, pygame.K_RALT]


        # Initialize other variables
        self.GamePlayer = Player()
        self.current_level_number = 1
        self.current_spawner_id = 0
        self.ended_game = False
        self.current_score = 0

        # Initialize clock
        self.clock = pygame.time.Clock()
        self.episode_clock = 0

        # Build the level
        self.Level = Map(self.current_level_number)


        # Define observation space (screen pixels)
        if self.env_rep_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(int(SCREEN_WIDTH/RESCALE_FACTOR), int(SCREEN_HEIGHT/RESCALE_FACTOR), 1), dtype=np.uint8)
        elif self.env_rep_type == 'text':
            box_shape = (1,len(self.Level.node_matrix), len(self.Level.node_matrix[0][:19]))
            all_possible_labels = ['scenery', 'tree', 'pinkpipe', 'door', 'items', 'trophy', 'player_spawner', 'tunnel', 'solid', 'water', 'tentacles', 'fire',
                                    'tentacles','gun','jetpack','moonstars','player']
            print(box_shape)
            if self.policy == 'CNN':
                self.observation_space = spaces.Box(low=0, high=len(all_possible_labels), shape=box_shape, dtype=np.uint8)
            else:
                self.observation_space = spaces.Box(low=0, high=len(all_possible_labels),shape=(box_shape[1]*box_shape[2],),dtype=np.uint8)
            print(self.observation_space)
            

        # Initialize screen and player positions
        self.player_position_x, self.player_position_y = self.Level.initPlayerPositions(self.current_spawner_id, self.GamePlayer)
        spawner_pos_x = self.Level.getPlayerSpawnerPosition(self.current_spawner_id)[0]
        self.game_screen.setXPosition(spawner_pos_x - 10, self.Level.getWidth())

        # UI Inits
        self.score_ui = 0  # Initial score. Every time it changes, update the UI
        self.jetpack_ui = False

        # Initialize other sprites
        self.death_timer = -1

        # Level processing controller
        self.ended_level = False


        if self.env_rep_type == 'text':
             self.label_enc  = LabelEncoder()
             self.label_enc.fit(all_possible_labels)
             self.unqiue_set = set()
        


    def reset(self, **kwargs):
        # Reset player, level, and game state


        self.GamePlayer = Player()
        self.current_level_number = 1
        self.current_spawner_id = 0
        self.ended_game = False
        self.current_score = 0

        self.episode_clock = 0

        # Build the level
        self.Level = Map(self.current_level_number)

        # Initialize screen and player positions

        if not self.random_respawn:
            self.player_position_x, self.player_position_y = self.Level.initPlayerPositions(self.current_spawner_id, self.GamePlayer)
            self.last_player_position_x = self.player_position_x
            self.last_player_position_y = self.player_position_y
            spawner_pos_x = self.Level.getPlayerSpawnerPosition(self.current_spawner_id)[0]
            self.game_screen.setXPosition(spawner_pos_x - 10, self.Level.getWidth())
        else:
            while True:
                y = np.random.randint(1,11)
                x = np.random.randint(1,19)
                if self.Level.node_matrix[y][x].getId() == "scenery":
                    spawner_pos_x = x
                    spawner_pos_y = y 
                    break
            
            self.GamePlayer.setCurrentState(STATE.BLINK)
            self.GamePlayer.setDirectionX(DIRECTION.IDLE)
            self.player_position_x = WIDTH_OF_MAP_NODE * spawner_pos_x
            self.player_position_y = HEIGHT_OF_MAP_NODE * spawner_pos_y
            
            self.last_player_position_x = self.player_position_x
            self.last_player_position_y = self.player_position_y
            self.game_screen.setXPosition(spawner_pos_x - 10, self.Level.getWidth())


        # UI Inits
        self.score_ui = 0  # Initial score. Every time it changes, update the UI
        self.jetpack_ui = False

        # Initialize other sprites
        self.death_timer = -1

        # Level processing controller
        self.ended_level = False
        

        return self._get_observation(), {}

    def step(self, action):
        # Convert action into player movement
        key_map = [0, 0, 0, 0]
        key_map[action] = 1
        self.GamePlayer.movementInput(key_map)

        # Run one game step
        self._run_game_step()
        self.GamePlayer.clearXMovement()

        self.episode_clock += 1
        self.clock.tick()

        # Return observation, reward, done, info
        # print("Time:", self.episode_clock)
        return self._get_observation(), self._get_reward(), self.ended_game, self.episode_clock >= 2048, {}

    def render(self, mode='human'):
        # Render the game screen

        # update UI
        self.game_screen.printOverlays(self.ui_tileset)
        self.game_screen.printUi(self.ui_tileset, self.GamePlayer, self.current_level_number)
        
        if not self.ended_level:
            if self.GamePlayer.inventory["gun"] == 1:
                self.game_screen.updateUiGun(self.ui_tileset)
            if self.GamePlayer.inventory["jetpack"] == 1 or self.jetpack_ui :
                self.game_screen.updateUiJetpack(self.ui_tileset, self.GamePlayer.inventory["jetpack"])
                self.jetpack_ui = True
            if self.GamePlayer.inventory["trophy"] == 1:
                self.game_screen.updateUiTrophy(self.ui_tileset)
                
        
        if self.score_ui != self.GamePlayer.score:
            self.game_screen.updateUiScore(self.GamePlayer.score, self.ui_tileset)
            self.score_ui = self.GamePlayer.score                
            
        pygame.display.update()

    def close(self):
        pygame.quit()

    def _run_game_step(self):
        # Get keys (inventory)
        # for event in pygame.event.get():
        #     # Stop moving
        #     if event.type == pygame.KEYUP:
        #         # Horizontally
        #         if event.key in [pygame.K_LEFT, pygame.K_RIGHT] and self.GamePlayer.getCurrentState() in [STATE.WALK, STATE.FLY, STATE.JUMP, STATE.CLIMB]:
        #             self.GamePlayer.clearXMovement()
        #         # Vertically
        #         elif event.key in [pygame.K_UP, pygame.K_DOWN] and self.GamePlayer.getCurrentState() in [STATE.FLY, STATE.CLIMB]:
        #             self.GamePlayer.setVelocityY(0)
        #     # Hit a key
        #     elif event.type == pygame.KEYDOWN:
        #         # Quit game
        #         if event.key == pygame.K_ESCAPE:
        #             self.ended_game = True
        #             ended_level = True
        #         # Use something from the inventory
        #         elif event.key in self.inv_keys:
        #             pass
        
        # # Get keys (movement)
        # pressed_keys = pygame.key.get_pressed()
        # key_map = [0, 0, 0, 0]
        # for i, key in enumerate(self.movement_keys):
        #     if pressed_keys[key]:
        #         key_map[i] = 1
        # self.GamePlayer.movementInput(key_map)

        # Update the player position in the level and treat collisions
        if self.GamePlayer.getCurrentState() != STATE.DESTROY:
            self.player_position_x, self.player_position_y = self.GamePlayer.updatePosition(self.player_position_x, self.player_position_y, self.Level, self.game_screen.getUnscaledHeight())

        # If the player ended the level, go on to the next
        if self.GamePlayer.getCurrentState() == STATE.ENDMAP:
            self.ended_level = True
            self.ended_game = True
            self.GamePlayer.setScore(self.GamePlayer.getScore() + END_LEVEL_SCORE)
            return
        # If the player died, spawn death puff and respawn player (if enough lives)
        elif self.GamePlayer.getCurrentState() == STATE.DESTROY:
            if death_timer == -1:
                self.GamePlayer.takeLife()
                DeathPuff = AnimatedTile("explosion", 0)
                death_timer = 120

            self.player_position_y += 0.25
            death_timer -= 1

            if death_timer == 0:
                death_timer = -1
                self.game_screen.setXPosition(self.Level.getPlayerSpawnerPosition(self.current_spawner_id)[0] - 10, self.Level.getWidth())
                del DeathPuff

                if self.GamePlayer.resetPosAndState() != -1:
                    self.player_position_x, self.player_position_y = self.Level.getPlayerSpawnerPosition(self.current_spawner_id)
                    self.player_position_x *= WIDTH_OF_MAP_NODE
                    self.player_position_y *= HEIGHT_OF_MAP_NODE
                else:
                    self.ended_level = True
                    self.ended_game = True

        # If the player is close enough to one of the screen boundaries, move the screen.
        player_close_to_left_boundary = (self.player_position_x <= self.game_screen.getXPositionInPixelsUnscaled() + BOUNDARY_DISTANCE_TRIGGER)
        player_close_to_right_boundary = (self.player_position_x >= self.game_screen.getXPositionInPixelsUnscaled() + self.game_screen.getUnscaledWidth() - BOUNDARY_DISTANCE_TRIGGER)
        reached_level_left_boundary = (self.game_screen.getXPosition() <= 0)
        reached_level_right_boundary = (self.game_screen.getXPosition() + self.game_screen.getWidthInTiles() > self.Level.getWidth())

        # Move screen left
        if player_close_to_left_boundary and not reached_level_left_boundary:
            self.game_screen.moveScreenX(self.Level, -15, self.tileset, self.ui_tileset, self.GamePlayer, self.current_level_number)
        # Move screen right
        elif player_close_to_right_boundary and not reached_level_right_boundary:
            self.game_screen.moveScreenX(self.Level, 15, self.tileset, self.ui_tileset, self.GamePlayer, self.current_level_number)
        # Not moving (just update the screen)
        else:
            self.game_screen.printMap(self.Level, self.tileset)

            if self.GamePlayer.getCurrentState() != STATE.DESTROY:
                # Print player accordingly to screen shift
                self.game_screen.printPlayer(self.GamePlayer, self.player_position_x - self.game_screen.getXPositionInPixelsUnscaled(), self.player_position_y, self.tileset)
            elif not self.ended_game:
                # Print death puff accordingly to screen shift
                self.game_screen.printTile(self.player_position_x - self.game_screen.getXPositionInPixelsUnscaled(), self.player_position_y, DeathPuff.getGraphic(self.tileset))
    
    def _get_text_representation(self):
       
        map_text_rep = []
        for line_index,node_line in enumerate(self.Level.node_matrix):
            map_text_rep.append(self.label_enc.transform(list(map(lambda x: x.getId(),node_line[:19]))))
        
        map_text_rep = np.array(map_text_rep,dtype=np.uint8)
        try:
            map_text_rep[int(self.player_position_y//HEIGHT_OF_MAP_NODE),int(self.player_position_x//WIDTH_OF_MAP_NODE)] = self.label_enc.transform(['player'])[0]
        except Exception as e:
            map_text_rep[0,0] = self.label_enc.transform(['player'])[0]
        return np.expand_dims(map_text_rep,0)
        

    def _get_observation(self):
        # Capture the current game screen
        if self.env_rep_type == 'image':

            game_surface = pygame.display.get_surface()

            game_surface = pygame.transform.scale(game_surface, (int(SCREEN_WIDTH/RESCALE_FACTOR), int(SCREEN_HEIGHT/RESCALE_FACTOR)))
            # Convert the game surface to a numpy array
            game_data = pygame.surfarray.array3d(game_surface)


            # Convert the color space from RGB to grayscale if needed
            game_data = np.dot(game_data[..., :3], [0.299, 0.587, 0.114])

            # Resize the game data if needed
            game_data = game_data.reshape((int(SCREEN_WIDTH/RESCALE_FACTOR), int(SCREEN_HEIGHT/RESCALE_FACTOR), 1))

            # Normalize the game data if needed
            # game_data = game_data / 255.0
            if self.policy == 'MLP':
                game_data = game_data.flatten()

        
    
        elif self.env_rep_type == 'text':
            game_data = self._get_text_representation()
            if self.policy == 'MLP':
                game_data = game_data.flatten()
       
        return game_data

    def _get_reward(self):
        # Calculate and return the reward based on game state
        # This can be customized based on the game's reward structure
        reward = self.GamePlayer.getScore() - self.current_score
        self.current_score = self.GamePlayer.getScore()
    
        if reward == 0:
            reward = -1
    
        if (self.last_player_position_x == self.player_position_x) and (self.last_player_position_y==self.player_position_y) and (reward==-1):
            reward = -2

        return reward

# Test the environment
if __name__ == '__main__':
    for i in range(100):
        env = DangerousDaveEnv(env_rep_type='text',random_respawn=True)
        obs,info = env.reset()
        for i in range(500):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            env.render()
            if done:
                print("Episode finished after {} timesteps".format(i+1))
                break
        env.close()
