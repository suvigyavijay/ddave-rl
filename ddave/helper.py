import pygame
from os import listdir, path
from os.path import isfile, join
from ddave.utils import *

'''
Tile and gfxs
'''

## split a string separating numbers from letters
def splitStringIntoLettersAndNumbers(string):
    split_string = []
    sub_string = ""
    index = 0

    while index < len(string):
        if string[index].isalpha():
            while index < len(string) and string[index].isalpha():
                sub_string += string[index]
                index += 1
        elif string[index].isdigit():
             while index < len(string) and string[index].isdigit():
                sub_string += string[index]
                index += 1
        else:
            index += 1
        split_string.append(sub_string)
        sub_string = ""

    return split_string


## get name and size properties from filename
def graphicPropertiesFromFilename(filename):
    split_filename = splitStringIntoLettersAndNumbers(filename)

    name = split_filename[0]
    height = int(split_filename[3])
    width = int(split_filename[1])

    return (name, height, width)

## returns dictionary
def load_game_tiles():
    game_tile_loc = path.join(path.dirname(path.abspath(__file__)), "tiles", "game/")
    game_tiles = [file for file in listdir(game_tile_loc) if isfile(join(game_tile_loc, file))] #load all the image files within the directory
    ui_tile_loc = path.join(path.dirname(path.abspath(__file__)), "tiles", "ui/")
    ui_tiles = [file for file in listdir(ui_tile_loc) if isfile(join(ui_tile_loc, file))]

    game_tile_dict = {} #init dictionary
    ui_tile_dict = {}
    
    #save game tiles
    for savedfile in game_tiles:
        image = pygame.image.load(game_tile_loc + savedfile).convert_alpha()

        tile_name, tile_height, tile_width = graphicPropertiesFromFilename(savedfile)

        game_tile_dict[tile_name] = (image, tile_height, tile_width)

    #save ui tiles
    for savedfile in ui_tiles:
        image = pygame.image.load(ui_tile_loc + savedfile).convert_alpha()

        tile_name, tile_height, tile_width = graphicPropertiesFromFilename(savedfile)

        ui_tile_dict[tile_name] = (image, tile_height, tile_width)        
        
    return game_tile_dict, ui_tile_dict

'''
Interpic
'''

def showTitleScreen(screen, tileset, ui_tiles):
    clock = pygame.time.Clock()
    
    # init graphics
    started_game = False
    titlepic_level = Map(1)
    dave_logo = AnimatedTile("davelogo", 0)
    overlay = Scenery("blacktile", 0)
    
    # clear screen on entering
    screen.clearScreen()
    
    # messages
    creator_text = "RECREATED BY ARTHUR, CATTANI AND MURILO"
    professor_text = "PROFESSOR LEANDRO K. WIVES"
    instr1_text = "PRESS SPACE TO START"
    instr2_text = "PRESSING ESC AT ANY MOMENT EXITS"
    
    while not started_game:
        pygame.display.update()
        
        # print level and tiles
        screen.setXPosition(14, titlepic_level.getWidth())
        screen.printMap(titlepic_level, tileset)
        screen.printTitlepicBorder(tileset)
        screen.printTile(104, 0, dave_logo.getGraphic(ui_tiles))   
        screen.printTile(0, BOTTOM_OVERLAY_POS, overlay.getGraphic(ui_tiles))
        
        # print text in center
        screen.printTextAlignedInCenter(creator_text, 47)
        screen.printTextAlignedInCenter(professor_text, 55)
        screen.printTextAlignedInCenter(instr1_text, BOTTOM_OVERLAY_POS + 2)
        screen.printTextAlignedInCenter(instr2_text, BOTTOM_OVERLAY_POS + 11)
        
        # if player pressed escape, exit game; space, start game
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                started_game = True  
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True    # return 0 so we know player pressed escape

        pygame.display.flip()
        clock.tick(200)
        
    # clear screen on exiting
    screen.clearScreen()
        
    return False

def showInterpic(completed_levels, screen, GamePlayer, tileset, ui_tileset):
    clock = pygame.time.Clock()
    
    # init graphics
    interpic_level = Map("interpic")
    screen.setXPosition(0, interpic_level.getWidth())    
    screen.printMap(interpic_level, tileset)
    screen.clearBottomUi(ui_tileset)
    
    # init player
    (player_absolute_x, player_absolute_y) = interpic_level.initPlayerPositions(0, GamePlayer)
    GamePlayer.setCurrentState(STATE.WALK)
    GamePlayer.setDirectionX(DIRECTION.RIGHT)
    GamePlayer.setSpriteDirection(DIRECTION.RIGHT)

    # init messages
    intertext = "GOOD WORK! ONLY " + str(NUM_OF_LEVELS - completed_levels + 1) + " MORE TO GO!"
    last_level_text = "THIS IS THE LAST LEVEL!!!"
    finish_text = "YES! YOU FINISHED THE GAME!!"
    
    # keep moving the player right, until it reaches the screen boundary
    player_reached_boundary = (player_absolute_x >= screen.getUnscaledWidth())

    while not player_reached_boundary:
        # if player pressed escape, quit game
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True # return so we treat exiting externally
                
        # update player pos and animation
        player_absolute_x = GamePlayer.movePlayerRight(player_absolute_x)
        GamePlayer.updateAnimator()
        
        # update screen
        screen.printMap(interpic_level, tileset)
        screen.printOverlays(ui_tileset)
        screen.printUi(ui_tileset, GamePlayer, completed_levels-1)
        screen.printPlayer(GamePlayer, player_absolute_x, player_absolute_y, tileset)
        
        # print text accordingly to the number of completed levels
        if completed_levels == NUM_OF_LEVELS + 1:
            screen.printTextAlignedInCenter(finish_text, 54)
        elif completed_levels == NUM_OF_LEVELS:
            screen.printTextAlignedInCenter(last_level_text, 54)
        else:
            screen.printTextAlignedInCenter(intertext, 54)

        player_reached_boundary = (player_absolute_x >= screen.getUnscaledWidth())
        
        pygame.display.flip()
        clock.tick(200)
        
    return False

def showWarpZone(completed_levels, screen, GamePlayer, tileset, ui_tileset):
    clock = pygame.time.Clock()
    
    # init graphics
    warp_level = Map("warp")
    screen.setXPosition(0, warp_level.getWidth())    
    screen.printMap(warp_level, tileset)
    screen.clearBottomUi(ui_tileset)
    
    # init player
    (player_absolute_x, player_absolute_y) = warp_level.initPlayerPositions(0, GamePlayer)
    GamePlayer.resetPosAndState()
    GamePlayer.setFallingState()
    GamePlayer.setGfxId(0)
    
    # keep moving the player right, until it reaches the screen boundary
    player_reached_bottom = (player_absolute_y >= screen.getUnscaledHeight())

    while not player_reached_bottom:
        # if player pressed escape, quit game
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True # return so we treat exiting externally
        
        player_absolute_y += 0.5
        
        # update screen
        screen.printMap(warp_level, tileset)        
        screen.printPlayer(GamePlayer, player_absolute_x, player_absolute_y, tileset)
        screen.printOverlays(ui_tileset)
        screen.printUi(ui_tileset, GamePlayer, completed_levels-1)

        player_reached_bottom = (player_absolute_y >= screen.getUnscaledHeight())
        
        pygame.display.flip()
        clock.tick(200)
        
    return False
    
    
def getBonusMapping(current_level):
    if current_level == 2: return 6
    elif current_level == 5: return 2
    elif current_level == 6: return 9
    elif current_level == 7: return 10
    elif current_level == 8: return 6
    elif current_level == 9: return 7
    elif current_level == 10: return 1
    elif current_level == 1: return 11
    else: return 1
    
def showScores(screen, tileset):
    pass
    
def savePlayerScore(player_score, screen, tileset):
    pass
        
def showCreditsScreen(screen, tileset):
    pass
   