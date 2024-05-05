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

RESCALE_FACTOR = config.getint('GAME', 'RESCALE_FACTOR')
END_LEVEL_SCORE = config.getint('GAME', 'END_LEVEL_SCORE')
EPISODE_TIMESTEPS = config.getint('GAME', 'EPISODE_TIMESTEPS')

STICKY_ACTIONS = config.getboolean('GAME', 'STICKY_ACTIONS')
LOCKED_DOOR = config.getboolean('GAME', 'LOCKED_DOOR')
STEP_PENALTY = config.getfloat('GAME', 'STEP_PENALTY')
CURRENT_LEVEL = config.getint('GAME', 'CURRENT_LEVEL')

PLAYER_RANDOM_SPAWN = config.getboolean('GAME', 'PLAYER_RANDOM_SPAWN')

class DangerousDaveEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode="human", env_rep_type="text"):
        self.render_mode = render_mode
        self.env_rep_type = env_rep_type

        # Initialize pygame
        pygame.init()
        self.game_screen = Screen(SCREEN_WIDTH, SCREEN_HEIGHT)

        # Initialize tiles
        self.tileset, self.ui_tileset = load_game_tiles()

        # Define action space (movement keys)
        self.action_space = spaces.Discrete(7)  # Up, Left, Right, Down, Up+Left, Up+Right, NoOp

        self.movement_keys = [pygame.K_UP, pygame.K_LEFT, pygame.K_RIGHT, pygame.K_DOWN]
        self.inv_keys = [pygame.K_LCTRL, pygame.K_RCTRL, pygame.K_LALT, pygame.K_RALT]

        # Initialize other variables
        self.GamePlayer = Player()
        self.current_level_number = CURRENT_LEVEL
        self.current_spawner_id = 0
        self.ended_game = False
        self.current_score = 0

        # Initialize clock
        self.clock = pygame.time.Clock()
        self.episode_clock = 0

        # Build the level
        self.Level = Map(self.current_level_number)

        # Initialize screen and player positions
        self.player_position_x, self.player_position_y = self.Level.initPlayerPositions(self.current_spawner_id, self.GamePlayer)
        spawner_pos_x = self.Level.getPlayerSpawnerPosition(self.current_spawner_id)[0]
        self.game_screen.setXPosition(spawner_pos_x - 10, self.Level.getWidth())
        if PLAYER_RANDOM_SPAWN:
            self.player_position_x, self.player_position_y = self.Level.getRandomPlayerSpawnerPosition()

        # UI Inits
        self.score_ui = 0  # Initial score. Every time it changes, update the UI
        self.jetpack_ui = False

        # Initialize other sprites
        self.death_timer = -1
        self.DeathPuff = AnimatedTile("explosion", 0)

        # Level processing controller
        self.ended_level = False
        
        # Define observation space (screen pixels)
        box_shape = (len(self.Level.node_matrix), len(self.Level.node_matrix[0][:19]))
        all_possible_labels = ['scenery', 'tree', 'pinkpipe', 'door', 'items', 'trophy',
                                'player_spawner', 'tunnel', 'solid', 'water', 'tentacles', 'fire',
                                'tentacles', 'gun', 'jetpack', 'moonstars', 'player']
        
        if self.env_rep_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(int(SCREEN_WIDTH/RESCALE_FACTOR), int(SCREEN_HEIGHT/RESCALE_FACTOR)), dtype=np.float32)
        elif self.env_rep_type == 'text':
            self.observation_space = spaces.Box(low=0, high=len(all_possible_labels)-1,shape=(box_shape[0] * box_shape[1],), dtype=np.uint8)
        elif self.env_rep_type == 'grid':
            self.observation_space = spaces.Box(low=0, high=len(all_possible_labels)-1,shape=(box_shape), dtype=np.uint8)
        
        # Text representation
        if self.env_rep_type == 'text' or self.env_rep_type == 'grid':
            self.label_enc  = LabelEncoder()
            self.label_enc.fit(all_possible_labels)
            self.unqiue_set = set()


    def reset(self, **kwargs):
        # Reset player, level, and game state
        self.GamePlayer = Player()
        self.current_level_number = CURRENT_LEVEL
        self.current_spawner_id = 0
        self.ended_game = False
        self.current_score = 0
        self.episode_clock = 0
        
        self._load_level()
        
        # UI Inits
        self.score_ui = 0  # Initial score. Every time it changes, update the UI
        self.jetpack_ui = False

        # Initialize other sprites
        self.death_timer = -1

        return self._get_observation(), {}
    
    def _load_level(self):
        self.GamePlayer.clearInventory()
        
        # Build the level
        self.Level = Map(self.current_level_number)

        # Initialize screen and player positions
        self.player_position_x, self.player_position_y = self.Level.initPlayerPositions(self.current_spawner_id, self.GamePlayer)
        spawner_pos_x = self.Level.getPlayerSpawnerPosition(self.current_spawner_id)[0]
        self.game_screen.setXPosition(spawner_pos_x - 10, self.Level.getWidth())
        if PLAYER_RANDOM_SPAWN:
            self.player_position_x, self.player_position_y = self.Level.getRandomPlayerSpawnerPosition()

        # Level processing controller
        self.ended_level = False

    def step(self, action):
        # Stop moving
        if action in [1, 2, 4, 5, 6] and self.GamePlayer.getCurrentState() in [STATE.WALK, STATE.FLY, STATE.JUMP, STATE.CLIMB]:
            self.GamePlayer.clearXMovement()
        if action in [0, 3, 4, 5, 6] and self.GamePlayer.getCurrentState() in [STATE.FLY, STATE.CLIMB]:
            self.GamePlayer.setVelocityY(0)
        
        # Convert action into player movement
        key_map = [0, 0, 0, 0]
        if action <= 3:
            key_map[action] = 1
        elif action == 4 or action == 5:
            key_map[action-3] = 1
            key_map[0] = 1
        else:
            pass
        self.GamePlayer.movementInput(key_map)

        # Run one game step
        if not STICKY_ACTIONS:
            self._run_game_step()
            self.clock.tick()
        else:
            for _ in range(16):
                if self.ended_level or self.episode_clock >= EPISODE_TIMESTEPS:
                    break
                self._run_game_step()
                self.clock.tick()

        self.episode_clock += 1

        # Return observation, reward, done, info
        return self._get_observation(), self._get_reward(), self.ended_game, self.episode_clock >= EPISODE_TIMESTEPS, {}

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
        
        # Update the player position in the level and treat collisions
        if self.GamePlayer.getCurrentState() != STATE.DESTROY:
            self.player_position_x, self.player_position_y = self.GamePlayer.updatePosition(self.player_position_x, self.player_position_y, self.Level, self.game_screen.getUnscaledHeight())

        # If the player ended the level, go on to the next
        if self.GamePlayer.getCurrentState() == STATE.ENDMAP:
            self.ended_level = True
            if LOCKED_DOOR:
                self.GamePlayer.setScore(self.GamePlayer.getScore() + END_LEVEL_SCORE)
            else:
                if self.GamePlayer.inventory["trophy"] == 1:
                    self.GamePlayer.setScore(self.GamePlayer.getScore() + END_LEVEL_SCORE)
                else:
                    self.GamePlayer.setScore(self.GamePlayer.getScore() - END_LEVEL_SCORE)
                    
            if self.ended_level:
                # Onto the next level
                self.current_level_number += 1
                    
                if self.current_level_number > NUM_OF_LEVELS and self.ended_level and not self.ended_game:
                    self.ended_game = True
                else:
                    self._load_level()
            return
        # If the player died, spawn death puff and respawn player (if enough lives)
        elif self.GamePlayer.getCurrentState() == STATE.DESTROY:
            if self.death_timer == -1:
                self.GamePlayer.takeLife()
                self.death_timer = 120

            self.player_position_y += 0.25
            self.death_timer -= 1

            if self.death_timer == 0:
                self.death_timer = -1
                self.game_screen.setXPosition(self.Level.getPlayerSpawnerPosition(self.current_spawner_id)[0] - 10, self.Level.getWidth())

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
                self.game_screen.printTile(self.player_position_x - self.game_screen.getXPositionInPixelsUnscaled(), self.player_position_y, self.DeathPuff.getGraphic(self.tileset))
    
    def _get_text_representation(self):
        map_text_rep = []
        for _, node_line in enumerate(self.Level.node_matrix):
            map_text_rep.append(self.label_enc.transform(list(map(lambda x: x.getId(), node_line[:19]))))
        
        map_text_rep = np.array(map_text_rep, dtype=np.uint8)
        map_text_rep[round(self.player_position_y / HEIGHT_OF_MAP_NODE), round(self.player_position_x / WIDTH_OF_MAP_NODE)] = self.label_enc.transform(['player'])[0]
        
        return map_text_rep
        
    def _get_image_representation(self):
        # Capture the current game screen
        game_surface = pygame.display.get_surface()

        game_surface = pygame.transform.scale(game_surface, (int(SCREEN_WIDTH/RESCALE_FACTOR), int(SCREEN_HEIGHT/RESCALE_FACTOR)))
        # Convert the game surface to a numpy array
        game_data = pygame.surfarray.array3d(game_surface)

        # Convert the color space from RGB to grayscale if needed
        game_data = np.dot(game_data[..., :3], [0.299, 0.587, 0.114])

        # Resize the game data if needed
        game_data = game_data.reshape((int(SCREEN_WIDTH/RESCALE_FACTOR), int(SCREEN_HEIGHT/RESCALE_FACTOR)))

        # Normalize the game data if needed
        # game_data = game_data / 255.0
        
        # convert to float32
        game_data = game_data.astype(np.float32)

        return game_data
        
    def _get_observation(self):
        if self.env_rep_type == 'image':
            return self._get_image_representation()
        elif self.env_rep_type == 'text':
            return self._get_text_representation().flatten()
        elif self.env_rep_type == 'grid':
            return self._get_text_representation()

    def _get_reward(self):
        # Calculate and return the reward based on game state
        # This can be customized based on the game's reward structure
        reward = self.GamePlayer.getScore() - self.current_score
        self.current_score = self.GamePlayer.getScore()
        
        if reward == 0:
            reward = STEP_PENALTY
        
        return reward

# Test the environment
if __name__ == '__main__':
    env = DangerousDaveEnv(env_rep_type='image')
    obs, _ = env.reset()
    print("Observation shape: ", obs.shape)
    episode_reward = 0
    for i in range(10000):
        print("Step: ", i)
        action = 6
        # wait for user input
        event = pygame.event.wait(100)
        # take input from the user from pygame
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            pygame.quit()
            break
        else:
            pressed_keys = pygame.key.get_pressed()
            if pressed_keys[pygame.K_UP] and pressed_keys[pygame.K_LEFT]:
                action = 4
            elif pressed_keys[pygame.K_UP] and pressed_keys[pygame.K_RIGHT]:
                action = 5
            elif pressed_keys[pygame.K_UP]:
                action = 0
            elif pressed_keys[pygame.K_LEFT]:
                action = 1
            elif pressed_keys[pygame.K_RIGHT]:
                action = 2
            elif pressed_keys[pygame.K_DOWN]:
                action = 3
            
        # action = env.action_space.sample()
        # action: up, left, right, down, up+left, up+right, no-op
        if action == 0:
            print("Action: Up")
        elif action == 1:
            print("Action: Left")
        elif action == 2:
            print("Action: Right")
        elif action == 3:
            print("Action: Down")
        elif action == 4:
            print("Action: Up+Left")
        elif action == 5:
            print("Action: Up+Right")
        elif action == 6:
            print("Action: No-op")
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        env.render()
        
        if done:
            print("Episode finished after {} timesteps".format(i+1))
            print("Episode reward: ", episode_reward)
            break
    env.close()
