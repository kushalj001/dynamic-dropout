import numpy as np
import cv2
import re
import imageio

# Rendering setup:

RES = 64  # resolution of frames --> RESxRES
LENGTH = 20
TRAINSET_PORTION = 0.9
LABEL = 0  # labels: 0-motion; 1-person; 2-scenario (4 overall);
# FILTER_MOTIONS = [1, 4, 5]  # filter out dynamic videos
# FILTER_MOTIONS = [0, 2, 3]  # filter out static videos
FILTER_MOTIONS = []  # do not filter
FILTER_SCENARIOS = []  # do not filter scenarios
FILTER_PERSONS = []  # do not filter persons
BACKGROUND_THRESHOLD = 10  # for seq 10, the value of 10 renders 2000 videos overall
SAVE_VIDEOS = True

########################################################################################################################
########################################################################################################################
########################################################################################################################

np.random.seed(12)

videos = []
labels = []

motion_types = {'boxing': 0, 'running': 1, 'handclapping': 2, 'handwaving': 3, 'jogging': 4, 'walking': 5}

f = open("00sequences.txt", "r")

lines = f.readlines()

for line in lines:
    line_split = line.split()
    if not line_split:
        break
    print(line_split)
    # Load video into a numpy array
    fname = line_split[0] + '_uncomp'
    print(fname)
    cap = cv2.VideoCapture('raw_videos/'+fname+'.avi')
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video = np.empty((frameCount, RES, RES), np.dtype('uint8'))
    fc = 0
    ret = True
    print(cap)
    try:
        while fc < frameCount and ret:
            ret, frame = cap.read()
            # to 1-channel format
            #print(frame.shape)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # to black-white -- unused
            # (thresh, frame) = cv2.threshold(frame, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # resize to RESxRES
            video[fc] = cv2.resize(frame, (RES, RES))
            fc += 1
    except:
        print("Bad file", fname)
        continue

    # get labels
    regex_motion = re.search('_(.*)_d', fname)
    motion_label = motion_types[regex_motion.group(1)]
    regex_person = re.search('person(.*)_', fname)
    person_label = int(regex_person.group(1)[0:2])
    regex_scenario = re.search('ing_d(.*)_', fname)
    scenario_label = int(regex_scenario.group(1)[0:2])

    print("File name: ", fname)
    print("Scenario label: ", scenario_label)
    print("Motion label: ", motion_label, " meaning ", regex_motion.group(1))
    print("Person label: ", person_label)

    if (scenario_label in FILTER_SCENARIOS) or (motion_label in FILTER_MOTIONS) or (person_label in FILTER_PERSONS):
        print("Filtering: scenario=",scenario_label, " motion=",motion_label, " person=", person_label)
        continue

    for i in range(len(line_split)-2):
        str_range = line_split[i + 2]
        print("--- Range now: ", str_range)
        if str_range[-1] == ',':
            str_range = str_range[:-1]
        fromto = [int(s) for s in str_range.split('-') if s.isdigit()]
        # print(fromto)
        for j in range(fromto[0], fromto[1], LENGTH):
            if j+LENGTH <= fromto[1]:

                # find corresponding video snippet and normalize it to [0,1]
                processed_video = video[j:j+LENGTH] / 255.0

                # record 3-channel video (it is still grayscale)
                if SAVE_VIDEOS:
                    writer = cv2.VideoWriter('processed_videos'+str(RES)+'/' + fname + str(j) + '-' + str(j + LENGTH) + '.avi',
                                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25, (RES, RES))
                    for k in range(processed_video.shape[0]):
                        writer.write(cv2.cvtColor((processed_video[k] * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR))
                    writer.release()

                # put channels to be the second dimension
                processed_video = np.expand_dims(processed_video, 1)
                print("Processed video shape: ", processed_video.shape)

                # Background-only threshold:
                if np.sum(processed_video[2] - processed_video[-2]) < BACKGROUND_THRESHOLD:
                    print("Background threshold activated !")
                    continue

                videos.append(processed_video)
                labels.append(np.array([motion_label, person_label, scenario_label]))

# Shuffle and store videos into numpy arrays
videos = np.array(videos)
labels = np.array(labels)
shuffler = np.random.permutation(videos.shape[0])
videos = videos[shuffler]
labels = labels[shuffler]
print("Total data matrix shape:", videos.shape)
print("Total label matrix shape:", labels.shape)

# split and save videos
train_samples = int(videos.shape[0] * TRAINSET_PORTION)
np.save('kth_train_data.npy', videos[:train_samples])
np.save('kth_train_labels.npy', labels[:train_samples, LABEL])
np.save('kth_valid_data.npy', videos[train_samples:])
np.save('kth_valid_labels.npy', labels[train_samples:, LABEL])

# save sample gifs
if SAVE_VIDEOS:
    import os
    os.makedirs("sample_gifs" + str(RES), exist_ok=True)
    for num in range(50):
        ok = False
        while not ok:
            imageio.mimwrite('sample_gifs' + str(RES) + '/sample' + str(num) + '.gif', videos[num, :, 0, :, :])
            try:
                imageio.mimwrite('sample_gifs'+str(RES)+'/sample' + str(num) + '.gif', videos[num, :, 0, :, :])
            except:
                print("rejected", num)
                continue
            print("accepted", num)
            ok = True
