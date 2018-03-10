import numpy as np
from brain_state_extraction import brain_st_extr


def main():
    # generate dictionary for subject and day filenames
    header = 'HOA112/dbrf/'
    subject = {}
    for s in range(20):
        s = s + 1
        subject[s] = header + 's' + str(s) + '/'

    days = {}
    for d in range(4):
        d = d + 1
        days[d] = 'day' + str(d) + '/'

    learn = {}
    for l in range(4):
        l = l + 1
        learn[l] = 'LEARN' + str(l) + '/'

    # creates condition matrix of HOA112 parcellated BOLD magnitudes
    g_val = 1
    brain_states_arr = np.empty(len(subject), dtype=tuple)
    condition_matrix = np.empty((len(days), len(learn), len(subject)), dtype=np.array)
    for s in subject:
        for d in days:
            for l in learn:
                # converts csv to numpy array
                condition_matrix = np.genfromtxt(subject[s] + days[d] + learn[l] +
                                                 'TS_HOA112.csv', delimiter=',')
        # stores returned tuple values from brain_state_extraction in brain_states array
        brain_states_arr[s-1] = brain_st_extr(condition_matrix, g_val)

    # extracts brain states from all subjects during day1 brain scans
    brain_states = np.array(brain_states_arr)
    for n in range(20):
        print('s' + str(n + 1) + ' = ' + str(brain_states[n]))


if __name__ == "__main__":
    main()

