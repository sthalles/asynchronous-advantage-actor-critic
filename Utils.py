from scipy import misc
import matplotlib.pyplot as plt

# returns a 84 x 84 image tensor as described in the deep minds paper
def process_input(img, width=84, height=84):
    out = img[:195, :] # get only the playing area of the image --> 195 for DemonAttack
    r, g, b = out[:,:,0], out[:,:,1], out[:,:,2]
    out = r * (299./1000.) + g * (587./1000.) + b * (114./1000.)
    out = misc.imresize(out, (width, height), interp="bilinear")
    return out

def display_transition(actions_names, memory):
    # display post states
    states = [memory[0]]
    id = 0
    f, axarr = plt.subplots(2, 4, figsize=(18,8))
    for state in states:
        print(state[:,:,0])
        for c in range(0,4):
            axarr[id][c].imshow(state[:,:,c], cmap=plt.cm.Greys);
            axarr[id][c].set_title('Action: ' + actions_names[memory[1]] + " Reward:" + str(memory[2]))
        id += 1
    plt.show()