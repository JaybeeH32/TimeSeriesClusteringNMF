import tslearn as ts
from tslearn.datasets import UCR_UEA_datasets
from dtw import dtw
import numpy as np

def load_ts_dataset(name):
    '''
    ['Adiac', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken', 'Car', 'CBF', 'ChlorineConcentration', 'CinCECGTorso', 'Coffee', 'Computers', 'CricketX', 'CricketY', 'CricketZ', 'DiatomSizeReduction', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxOutlineAgeGroup', 'DistalPhalanxTW', 'Earthquakes', 'ECG200', 'ECG5000', 'ECGFiveDays', 'ElectricDevices', 'FaceAll', 'FaceFour', 'FacesUCR', 'FiftyWords', 'Fish', 'FordA', 'FordB', 'GunPoint', 'Ham', 'HandOutlines', 'Haptics', 'Herring', 'InlineSkate', 'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7', 'Mallat', 'Meat', 'MedicalImages', 'MiddlePhalanxOutlineCorrect', 'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxTW', 'MoteStrain', 'NonInvasiveFatalECGThorax1', 'NonInvasiveFatalECGThorax2', 'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxTW', 'RefrigerationDevices', 'ScreenType', 'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves', 'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl', 'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG', 'TwoPatterns', 'UWaveGestureLibraryX', 'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'UWaveGestureLibraryAll', 'Wafer', 'Wine', 'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga', 'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions', 'CharacterTrajectories', 'Cricket', 'DuckDuckGeese', 'EigenWorms', 'Epilepsy', 'EthanolConcentration', 'ERing', 'FaceDetection', 'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat', 'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery', 'NATOPS', 'PenDigits', 'PEMS-SF', 'Phoneme', 'RacketSports', 'SelfRegulationSCP1', 'SelfRegulationSCP2', 'SpokenArabicDigits', 'StandWalkJump', 'UWaveGestureLibrary']
    '''
    
    uea_ucr = UCR_UEA_datasets()
    if name in uea_ucr.list_datasets():
        return uea_ucr.load_dataset(name)
    else:
        raise ValueError(f'Dataset name "{name}" is not correct.')


# dtw related functions
def compute_dtw(signal1, signal2, normalized=True):
    align = dtw(signal1, signal2, dist_method='euclidean')
    if normalized:
        return align.normalizedDistance
    else:
        return align.distance

def distance_matrix(dataset, normalized=True):
    N = len(dataset)
    dist_matrix = np.zeros(shape=(N, N))
    for i in range(N):
        for j in range(i + 1, N):
            dist_matrix[i, j] = compute_dtw(dataset[i], dataset[j], normalized)
            dist_matrix[j, i] = dist_matrix[i, j]
    return dist_matrix

def epsilon_graph_hard(distance_matrix, epsilon):
    return np.where(distance_matrix <= epsilon, distance_matrix, 0)

def epsilon_graph_mean(distance_matrix):
    threshold = np.mean(distance_matrix.flatten())
    return np.where(distance_matrix <= threshold, distance_matrix, 0)