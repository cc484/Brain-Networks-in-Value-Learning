import numpy as np
from brain_state_extraction import brain_st_extr


def main():
    # generate dictionary for subject filenames
    header = 'HOA112/dbrf/'
    subject = {}
    for n in range(20):
        n = n + 1
        subject[n] = header + 's' + str(n) + '/'

    # extracts brain states from all subjects during day1 brain scans
    g_val = 1
    brain_states_arr = []
    for s in subject:
        # converts csv to numpy array
        condition_matrix = np.genfromtxt(subject[s] + 'day1/LEARN1/TS_HOA112.csv', delimiter=',')
        # stores returned tuple values from brain_state_extraction in brain_states array
        brain_states_arr[s] = brain_st_extr(condition_matrix, g_val)

    brain_states = np.array(brain_states_arr)
    for n in range(20):
        return 's' + str(n + 1) + str(brain_states[n])


if __name__ == "__main__":
    main()

