import gymnasium as gym
import flappy_bird_gymnasium
import pygame


# Create Flappy Bird environment
env = gym.make("FlappyBird-v0", render_mode="human")

# Reset the environment
state, info = env.reset()

done = False

# Initialize pygame
pygame.init()

# Gym has already created the game window
screen = pygame.display.get_surface()


while not done:

    # 0 = do nothing
    # 1 = flap
    action = 0

    # Check keyboard events
    for event in pygame.event.get():

        # Close window
        if event.type == pygame.QUIT:
            done = True

        # Keyboard key pressed
        elif event.type == pygame.KEYDOWN:

            # Space = flap
            if event.key == pygame.K_SPACE:
                action = 1

    # Perform action
    state, reward, done, truncated, info = env.step(action)

    # Display game
    env.render()


# Close everything
env.close()
pygame.quit()