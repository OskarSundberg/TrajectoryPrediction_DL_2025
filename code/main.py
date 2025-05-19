import torch
import gc
import os

from classes.location import Location
from classes.visualization import Visualization
from classes.Experiments.experiment_manager import ExperimentManager

# From Valhallavagen_grid.png
Valhallavagen_X_min = -10.75
Valhallavagen_X_max = 10.75
Valhallavagen_Y_min = -38.27
Valhallavagen_Y_max = 38.27

Valhallavagen_env_vectors = {
    'Type': ['Lane', 'Lane', 'Lane', 'Lane', 'SideWalk', 'SideWalk', 'SideWalk', 'SideWalk', 'PreCrosswalk', 'PreCrosswalk', 'CrossWalk'],
    'Top_Left': [(225, 2), (289, 7), (174, 694), (227, 705), (195, 1), (121, 695), (355, 10), (295, 696), (138, 572), (313, 596), (184, 585)],
    'Top_Right': [(289, 7), (354, 12), (227, 705), (289, 707), (225, 2), (171, 707), (429, 15), (354, 703), (186, 585), (368, 599), (305, 603)],
    'Bottom_Left': [(193, 550), (245, 551), (55, 1486), (115, 1498), (140, 541), (1, 1471), (317, 547), (190, 1514), (129, 635), (300, 684), (177, 669)],
    'Bottom_Right': [(245, 551), (308, 558), (115, 1498), (186, 1513), (191, 545), (53, 1485), (377, 551), (267, 1530), (177, 641), (360, 689), (295, 684)],
    'Top_Left_to_Top_Right': [(64, 5), (65, 5), (53, 11), (62, 2), (30, 1), (50, 12), (74, 5), (59, 7), (48, 13), (55, 3), (121, 18)],
    'Top_Right_to_Bottom_Right': [(-44, 544), (-46, 546), (-112, 793), (-103, 806), (-34, 543), (-118, 778), (-52, 536), (-87, 827), (-9, 56), (-8, 90), (-10, 81)],
    'Bottom_Right_to_Bottom_Left': [(-52, -1), (-63, -7), (-60, -12), (-71, -15), (-51, -4), (-52, -14), (-60, -4), (-77, -16), (-48, -6), (-60, -5), (-118, -15)],
    'Bottom_Left_to_Top_Left': [(32, -548), (44, -544), (119, -792), (112, -793), (55, -540), (120, -776), (38, -537), (105, -818), (9, -63), (13, -88), (7, -84)]
}

valhallavagen_ENV = {
    'Type': [
        'Lane', 'Lane', 'Lane', 'Lane', 
        'SideWalk', 'SideWalk', 'SideWalk', 'SideWalk',
        'PreCrosswalk', 'PreCrosswalk', 'CrossWalk', 
        ],
    'Top_Left': [
        (225, 2), (289, 7), (174, 694), (227, 705), 
        (195, 1), (121, 695), (355, 10), (295, 696), 
        (138, 572), (313, 596), (184, 585) 
        ],
    'Top_Right': [
        (289, 7), (354, 12), (227, 705), (289, 707), 
        (225, 2), (171, 707), (429, 15), (354, 703),
        (186, 585), (368, 599), (305, 603)
        ],
    'Bottom_Left': [
        (193, 550), (245, 551), (55, 1486), (115, 1498), 
        (140, 541), (1, 1471), (317, 547), (190, 1514),
        (129, 635), (300, 684), (177, 669)
        ],
    'Bottom_Right': [
        (245, 551), (308, 558), (115, 1498), (186, 1513), 
        (191, 545), (53, 1485), (377, 551), (267, 1530),
        (177, 641), (360, 689), (295, 684)
        ]
}

valhallavagen_static_objs = {
    'Type': ['Pole', 'Pole', 'Pole', 'Pole'],
    'Top_Left': [(166, 674), (180, 573), (325, 583), (311, 690)],
    'Top_Right': [None] * 4,
    'Bottom_Left': [None] * 4,
    'Bottom_Right': [None] * 4
}

roi_valhallavagen = [
    {'roi_x_min':150, 'roi_x_max':225, 'roi_y_min':525, 'roi_y_max':725, 'location':'Left'},
    {'roi_x_min':275, 'roi_x_max':495, 'roi_y_min':500, 'roi_y_max':700, 'location':'Right'}
    ]

# From Torpagatan_grid.png
Torpagatan_X_min = -30.02 
Torpagatan_X_max = 30.02
Torpagatan_Y_min = -20
Torpagatan_Y_max = 20

Torpagatan_env_vectors = {
    'Type': ['Zebra Crossing', 'PreCrosswalk', 'PreCrosswalk', 'PreCrosswalk', 'PreCrosswalk', 'CrossWalk', 'Lane', 'Lane', 'Lane', 'Lane', 'Lane', 'Lane', 'SideWalk', 'SideWalk', 'SideWalk', 'SideWalk'],
    'Top_Left': [(428, 261), (126, 80), (386, 143), (434, 218), (391, 387), (253, 97), (482, 285), (464, 353), (124, 85), (94, 159), (259, 23), (318, 20), (50, 282), (450, 152), (391, 41), (446, 482)],
    'Top_Right': [(482, 285), (259, 23), (450, 152), (509, 228), (452, 409), (386, 143), (1293, 572), (1269, 625), (428, 261), (418, 329), (318, 20), (391, 41), (367, 442), (1340, 468), (446, 59), (1220, 736)],
    'Bottom_Left': [(391, 387), (202, 142), (384, 204), (428, 261), (337, 521), (202, 142), (464, 353), (452, 409), (94, 159), (70, 222), (253, 97), (311, 115), (30, 332), (509, 228), (386, 143), (458, 602)],
    'Bottom_Right': [(452, 409), (253, 97), (512, 224), (482, 285), (416, 549), (384, 204), (1269, 625), (1245, 679), (418, 329), (391, 387), (311, 115), (386, 143), (337, 521), (1320, 507), (450, 152), (1176, 831)],
    'Top_Left_to_Top_Right': [(54, 24), (133, -57), (64, 9), (75, 10), (61, 22), (133, 46), (811, 287), (805, 272), (304, 176), (324, 170), (59, -3), (73, 21), (317, 160), (890, 316), (55, 18), (774, 254)],
    'Top_Right_to_Bottom_Right': [(-30, 124), (-6, 74), (62, 72), (-27, 57), (-36, 140), (-2, 61), (-24, 53), (-24, 54), (-10, 68), (-27, 58), (-7, 95), (-5, 102), (-30, 79), (-20, 39), (4, 93), (-44, 95)],
    'Bottom_Right_to_Bottom_Left': [(-61, -22), (-51, 45), (-128, -20), (-54, -24), (-79, -28), (-182, -62), (-805, -272), (-793, -270), (-324, -170), (-321, -165), (-58, -18), (-75, -28), (-307, -189), (-811, -279), (-64, -9), (-718, -229)],
    'Bottom_Left_to_Top_Left': [(37, -126), (-76, -62), (2, -61), (6, -43), (54, -134), (51, -45), (18, -68), (12, -56), (30, -74), (24, -63), (6, -74), (7, -95), (20, -50), (-59, -76), (5, -102), (-12, -120)]
}

torpagatan_ENV = {
    'Type': [
        "Zebra Crossing", "PreCrosswalk", "PreCrosswalk", "PreCrosswalk", 
        "PreCrosswalk", "CrossWalk", "Lane", "Lane", "Lane", "Lane", 
        "Lane", "Lane", "SideWalk", "SideWalk", "SideWalk", "SideWalk"
        ],
    'Top_Left': [
        (428, 261), (126, 80), (386, 143), (434, 218), (391, 387), (253, 97), 
        (482, 285), (464, 353), (124, 85), (94, 159), (259, 23), (318, 20), 
        (50, 282), (450, 152), (391, 41), (446, 482)
        ],
    'Top_Right': [
        (482, 285), (259, 23), (450, 152), (509, 228), (452, 409), (386, 143), 
        (1293, 572), (1269, 625), (428, 261), (418, 329), (318, 20), (391, 41), 
        (367, 442), (1340, 468), (446, 59), (1220, 736)
        ],
    'Bottom_Left': [
        (391, 387), (202, 142), (384, 204), (428, 261), (337, 521), (202, 142),
        (464, 353), (452, 409), (94, 159), (70, 222), (253, 97), (311, 115), 
        (30, 332), (509, 228), (386, 143), (458, 602)
        ],
    'Bottom_Right': [
        (452, 409), (253, 97), (512, 224), (482, 285), (416, 549), (384, 204), 
        (1269, 625), (1245, 679), (418, 329), (391, 387), (311, 115),
        (386, 143), (337,521), (1320, 507), (450, 152), (1176, 831)
        ]
}

torpagatan_static_objs = {
    'Type': ["Tree", "Tree", "Tree", "Tree", "Sign", "Sign", "Sign"],
    'Top_Left': [(693, 629), (194, 342), (78, 261), (1047, 449), (500, 279), (380, 401), (439, 106)],
    'Top_Right': [None] * 7,
    'Bottom_Left': [None] * 7,
    'Bottom_Right': [None] * 7
}


roi_torpagatan = [
    {'roi_x_min':670, 'roi_x_max':750, 'roi_y_min':580, 'roi_y_max':675, 'object_x':693, 'object_y':630, 'type':'Tree'},
    {'roi_x_min':175, 'roi_x_max':225, 'roi_y_min':300, 'roi_y_max':365, 'object_x':194, 'object_y':340, 'type':'Tree'},
    {'roi_x_min':350, 'roi_x_max':450, 'roi_y_min':350, 'roi_y_max':450, 'object_x':380, 'object_y':401, 'type':'Sign'},
    {'roi_x_min':460, 'roi_x_max':560, 'roi_y_min':225, 'roi_y_max':305, 'object_x':500, 'object_y':279, 'type':'Sign'},
    {'roi_x_min':1020, 'roi_x_max':1090, 'roi_y_min':400, 'roi_y_max':490, 'object_x':1047, 'object_y':449, 'type':'Tree'}
    ]

def vizualize_location(exp_manager : ExperimentManager, model_name : str):
    def load_tensors(model : str, location : str):
        pred = torch.load(f"./data/Results/{model}/{location}/pred.pt", weights_only=True)
        tgt = torch.load(f"./data/Results/{model}/{location}/tgt.pt", weights_only=True)
        
        return {'pred': pred, 'tgt': tgt}

    tensors = load_tensors(model=model_name, location=exp_manager.location.location_name)

    exp_manager.viz.visualize_roi_valhallavagen(pred=tensors["pred"],
                                                tgt=tensors["tgt"],
                                                model_name=model_name,
                                                is_pred=False)

# def remove_directory_contents(directory_path):
#    # Ensure the directory exists
#    if os.path.exists(directory_path) and os.path.isdir(directory_path):
#        # Iterate over all files and directories in the specified directory
#        for item in os.listdir(directory_path):
#            item_path = os.path.join(directory_path, item)
#            # Remove files and directories
#            if os.path.isfile(item_path):
#                print(f"Deleting {item_path}")
#                os.remove(item_path)

# needed for when debugging on cpu
#print(f"cwd = {os.getcwd()}")
os.chdir("/home/sali20jt/viscando/TrajectoryPrediction_DL_2025/code")
#print(f"cwd = {os.getcwd()}")

if __name__=="__main__":
    # if False:
    #    remove_directory_contents("./data/CombinedData/Torpagatan")
    #    remove_directory_contents("./data/CombinedData/Valhallavagen")
    
    #    remove_directory_contents("./data/Datasets/Torpagatan/Test")
    #    remove_directory_contents("./data/Datasets/Torpagatan/Train")
    #    remove_directory_contents("./data/Datasets/Torpagatan/Val")
    
    #    remove_directory_contents("./data/Datasets/Valhallavagen/Test")
    #    remove_directory_contents("./data/Datasets/Valhallavagen/Train")
    #    remove_directory_contents("./data/Datasets/Valhallavagen/Val")

    # Check if GPU is available 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Apply deterministic on CUDA convoltion operations
    torch.backends.cudnn.deterministic = True
    # Disable benchmark mode
    torch.backends.cudnn.benchmark = False
    # Create a manual seed for testing
    torch.manual_seed(42)
    
    epochs = 1
    num_layers = 16
    num_heads = 8
    dropout = 0.3
    learning_rate = 0.000015
    src_len = 10
    tgt_len = 40
    batch_size = 32
    hidden_size = 512
    earlystopping = 30

    # epochs = 300
    # num_layers = 16
    # num_heads = 8
    # dropout = 0.3
    # learning_rate = 0.000015
    # src_len = 10
    # tgt_len = 40
    # batch_size = 32
    # hidden_size = 512
    # earlystopping = 30

    
    valhallavagen = Location(
        min_x=Valhallavagen_X_min, 
        max_x=Valhallavagen_X_max, 
        min_y=Valhallavagen_Y_min, 
        max_y=Valhallavagen_Y_max, 
        location_name="Valhallavagen",
        env_vectors=Valhallavagen_env_vectors,
        env_polygons=valhallavagen_ENV,
        static_objects=valhallavagen_static_objs,
        roi=roi_valhallavagen
        )
    viz = Visualization(valhallavagen)
    experiment_manager = ExperimentManager(
        location=valhallavagen, 
        visualization=viz, 
        epochs=epochs, 
        learning_rate=learning_rate, 
        num_layers=num_layers, 
        num_heads=num_heads, 
        dropout=dropout,
        src_len=src_len,
        tgt_len=tgt_len,
        batch_size=batch_size,
        hidden_size=hidden_size,
        earlystopping=earlystopping
        )
    
    #pyexperiment_manager.experiment_base()
    #experiment_manager.experiment_transformer()
    #experiment_manager.experiment_star()
    #experiment_manager.experiment_saestar()
    experiment_manager.experiment_seastar(device=device)

    #vizualize_location(exp_manager=experiment_manager, model_name="SEASTAR")

    del valhallavagen, viz, experiment_manager
    gc.collect()
    
    # torpagatan = Location(
    #     min_x=Torpagatan_X_min, 
    #     max_x=Torpagatan_X_max, 
    #     min_y=Torpagatan_Y_min, 
    #     max_y=Torpagatan_Y_max, 
    #     location_name="Torpagatan",
    #     env_vectors=Torpagatan_env_vectors,
    #     env_polygons=torpagatan_ENV,
    #     static_objects=torpagatan_static_objs,
    #     roi=roi_torpagatan
    #     )
    # viz = Visualization(torpagatan)
    # experiment_manager = ExperimentManager(
    #     location=torpagatan, 
    #     visualization=viz, 
    #     epochs=epochs, 
    #     learning_rate=learning_rate, 
    #     num_layers=num_layers, 
    #     num_heads=num_heads, 
    #     dropout=dropout,
    #     src_len=src_len,
    #     tgt_len=tgt_len,
    #     batch_size=batch_size,
    #     hidden_size=hidden_size,
    #     earlystopping=earlystopping
    #     )
    
    #experiment_manager.experiment_base()
    #experiment_manager.experiment_transformer()
    #experiment_manager.experiment_star()
    #experiment_manager.experiment_saestar()
    #experiment_manager.experiment_seastar()
