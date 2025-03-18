import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import copy

# degradate_feature altro non fa che moltiplicare un segnale per un coefficiente, a tempo t0 il coefficiente è
# pari a 1 e puo andare a 0 (attenuation) o fv (amplification) dove fv è massimo 2 (ovvero il segnale raddoppia)
# questa funzione prende un qualsiasi segnale e lo degrada in modo tale che a tempo t1 la degradazione parte
# e a partire da fv=1 il  coefficiente di amplificazione tempo t2 raggiunge il valore fv = final value
def degradate(sig, t1, t2, curr_t,** kwargs):
    mode_array = kwargs['mode_array']
    fv_array = kwargs['final_values_array']
    SIG = copy.deepcopy(sig)
    for i,mode in enumerate(mode_array):
        c_t1 = t1[i]
        c_t2 = t2[i]
        c_mode = mode_array[i]
        c_fv = fv_array[i]
        SIG[:,i]= degradate_feature(sig[:,i],c_t1,c_t2,curr_t, mode=c_mode, final_value=c_fv)
    return SIG
# degradate_featur

def degradate_feature(sig, t1, t2, curr_t,** kwargs):
    mode = kwargs['mode']
    final_value = kwargs['final_value']
    sigsig = copy.deepcopy(sig)

    fv = final_value
    if mode == "exp":
        if fv == 0:
            A = np.log(2) / (t2 - t1)
            B = np.exp(-A * t1)
            C = 2
            deg_fun = lambda t: C - B * np.exp(A * t)
        else:
            A = np.log(fv) / (t2 - t1)
            B = np.exp(-A * t1)
            C = 0
            deg_fun = lambda t: C + B * np.exp(A * t)
    elif mode == "log":
        A = (np.exp(fv - 1) - 1) / (t2 - t1)
        B = 1 - A * t1
        C = 1
        deg_fun = lambda t: C + np.log(A * t + B)
    elif mode == "lin":
        A = (fv - 1) / (t2 - t1)
        C = - A * t1 + 1
        deg_fun = lambda t: C + A * t
    elif mode == "pol":
        B = (fv - 1) / (t2 * t2 - t1 * t2)
        A = -t1 * B
        C = 1
        deg_fun = lambda t: C + A * t + B * t * t
    else:
        print("there is an error in the degradation mode text! Pleas correct")

    t_healty = np.arange(0, t1, 0.1)
    t_faulty = np.arange(t1, t2, 0.1)

    healty_coeff = np.ones(len(t_healty))
    faulty_coeff = deg_fun(t_faulty)

    if curr_t < t_healty[-1]:
        curr_coeff = 1
    else:
        curr_coeff = deg_fun(curr_t)
        curr_coeff = deg_fun(curr_t) + generate_gaussian_random_number(deg_fun(curr_t),0.005)
    #
    # time = np.concatenate((t_healty, t_faulty))
    # signal = np.concatenate((healty_coeff, faulty_coeff))

    return sigsig * curr_coeff


def plot_degradation_fun(signal, time, t1, t2, st, L_bound, U_bound):

    plt.plot(time, signal, label='Signal')

    # Plot vertical lines at t1 and t2
    plt.axvline(x=t1, color='r', linestyle='--', label='starting degradation time')
    plt.axvline(x=t2, color='b', linestyle='--', label='end of useefull life')

    # Plot horizontal lines at st, L_bound, and U_bound
    plt.axhline(y=st, color='g', linestyle='--', label='lower bound')
    plt.axhline(y=L_bound, color='m', linestyle='--', label='zero')
    plt.axhline(y=U_bound, color='y', linestyle='--', label='Upper bound')

    # Add labels and legend
    plt.xlabel('time')
    plt.ylabel('Degradation coefficient')
    plt.legend()

    # Show the plot
    plt.show()

def generate_color_variations_array(base_colors, N):
    # Get RGB values of the base color
    color_variations_list = list()
    for base_color in base_colors:
        base_rgb = mcolors.to_rgba(mcolors.CSS4_COLORS[base_color])[:3]

        # Generate N variations of the color by adjusting brightness
        color_variations = []
        for i in range(N):
            brightness = i / float(N - 1)
            variation_rgb = [c * brightness for c in base_rgb]
            color_variations.append(mcolors.to_hex(variation_rgb))
        color_variations_list.append(color_variations)
    return color_variations_list

def generate_color_variations(base_color, N):
    # Get RGB values of the base color
    base_rgb = mcolors.to_rgba(mcolors.CSS4_COLORS[base_color])[:3]

    # Generate N variations of the color by adjusting brightness
    color_variations = []
    for i in range(N):
        brightness = i / float(N - 1)
        variation_rgb = [c * brightness for c in base_rgb]
        color_variations.append(mcolors.to_hex(variation_rgb))

    return color_variations

# # Example usage:
# base_color = "blue"
# num_variations = 20
# color_variations = generate_color_variations(base_color, num_variations)
#
# print(f"Color Variations for {base_color}: {color_variations}")

def plot_colored_rectangles(color_variations):
    num_rectangles = len(color_variations)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot rectangles with corresponding colors
    for i, color in enumerate(color_variations):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))

    # Set x-axis labels
    ax.set_xticks(range(num_rectangles))
    ax.set_xticklabels(color_variations, rotation=45, ha='right')

    # Remove y-axis labels
    ax.set_yticks([])

    # Set plot title
    plt.title("Color Variations")

    # Show the plot
    plt.show()

import numpy as np

def generate_gaussian_random_number(mean, variance):
    # Generate a Gaussian random number with mean 0 and variance 1
    random_number = np.random.randn()

    # Scale the random number to have the desired variance
    scaled_random_number = np.sqrt(variance) * random_number

    # Add the desired mean
    gaussian_random_number = mean + scaled_random_number

    return gaussian_random_number


def transformation_1(signal):
    signal = signal *1
    return signal
def transformation_2(signal):
    signal = signal *1
    return signal


def transformation_N(signal):
    signal = signal *1
    return signal