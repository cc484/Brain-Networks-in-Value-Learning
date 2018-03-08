import numpy as np
from brain_state_extraction import brain_st_extr


def main():
    # generate dictionary for subject and day filenames
    header = 'HOA112/dbrf/'
    subject_dict = {}
    for s in range(20):
        s = s + 1
        subject_dict[s] = header + 's' + str(s) + '/'

    days_dict = {}
    for d in range(4):
        d = d + 1
        days_dict[d] = 'day' + str(d) + '/'

    learn_dict = {}
    for l in range(4):
        l = l + 1
        learn_dict[l] = 'LEARN' + str(l) + '/'

    # extracts brain states from all subjects during day1 brain scans
    g_val = 1
    brain_states_arr = np.empty(len(subject_dict), dtype=tuple())
    for s, d, l in subject_dict, days_dict, learn_dict
        # converts csv to numpy array
        condition_matrix = np.genfromtxt(subject_dict[s] + days_dict[d] + learn_dict[l] +
                                        'TS_HOA112.csv' , delimiter=',')
        for i in range(len(subject_dict)):
            # stores returned tuple values from brain_state_extraction in brain_states array
            brain_states_arr[i] = brain_st_extr(condition_matrix, g_val)

    brain_states = np.array(brain_states_arr)
    for n in range(20):
        print('s' + str(n + 1) + ' = ' + str(brain_states[n]))


if __name__ == "__main__":
    main()

