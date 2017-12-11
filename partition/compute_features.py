is_training = True
#regularization strength for the minimal partition
reg_strength = .2
k_nn_adj = 10
k_nn_desc_min = 15
k_nn_desc_max = 50
k_nn_desc_step = 5
d_se_max = 10
#root of the data directory
root = "/media/landrieuloic/Data/S3DIS/"
#list of subfolders to be processed
#areas = ["Area_1/","Area_2/","Area_3/","Area_4/","Area_5/","Area_6/"]
areas = ["Area_0/"]
num_area = len(areas)
if not os.path.isdir(root + "/clouds"):
    os.mkdir(root + "/clouds")
if not os.path.isdir(root + "/descriptors"):
    os.mkdir(root + "/descriptors")
if not os.path.isdir(root + "/superpoint_graphs"):
    os.mkdir(root + "/superpoint_graphs")
confusion_matrix = np.array([num_area, 1])
for i_area in range(0,num_area):
    area = areas[i_area]
    print("=================\n   "+area+"\n=================")
    raw_folder = root + "raw/"               +area
    ply_folder = root + "clouds/"            +area
    des_folder = root + "descriptors/"       +area
    spg_folder = root + "/superpoint_graphs/"+area
    if not os.path.isdir(raw_folder):
        raise ValueError("%s do not exists" % raw_folder)
    if not os.path.isdir(ply_folder):
        os.mkdir(ply_folder)
    if not os.path.isdir(des_folder):
        os.mkdir(des_folder)
    if not os.path.isdir(spg_folder):
        os.mkdir(spg_folder)
    rooms = [os.path.join(raw_folder, o) for o in os.listdir(raw_folder) 
                    if os.path.isdir(os.path.join(raw_folder,o))]
    if (len(rooms) == 0):
        raise ValueError('%s is empty' % raw_folder)
    n_rooms = len(rooms)
    i_room = 0
    rooms = [rooms[0]]
    for room in rooms:
        room_name = os.path.splitext(os.path.basename(room))[0]
        room_raw_folder = raw_folder      + room_name + '/'
        room_raw_path   = room_raw_folder + room_name + ".txt"
        room_ply_path   = ply_folder      + room_name + '.ply' 
        room_des_path   = des_folder      + room_name + '.h5'
        room_spg_path   = spg_folder      + room_name + '.h5' 
        i_room = i_room + 1
        print(str(i_room) + " / " + str(n_rooms) + "---> "+room_name)
        #--- build the labeled ply file ---
        if os.path.isfile(room_ply_path):
            print("reading the existing ply file...")
            xyz, rgb, room_labels, room_object_indices = read_ply(room_ply_path)
        else :
            print("creating the ply file...")
            xyz, rgb, room_labels, room_object_indices = get_objects(room_raw_path);
            write_ply(room_ply_path, room_ver, room_labels, room_object_indices)
        #--- build the descriptor h5 file ---
        if os.path.isfile(room_des_path):
            print("reading the existing descriptor file...")
            desc, graph_nn = read_descriptors(room_des_path)
        else :
            print("creating the feature file...")
            #---compute 10 nn graph-------
            graph_nn, target_desc = compute_graph_nn_2(xyz, k_nn_adj, k_nn_desc_max)
            #---compute descriptors-------
            desc = np.array(libdesc.compute_descriptors(xyz.tolist(), target_desc.tolist(), k_nn_desc_min, k_nn_desc_max, k_nn_desc_step), dtype = 'float32')
            del target_desc
            write_descriptors(room_des_path, desc, graph_nn)
        #--compute the partition------
        if os.path.isfile(room_spg_path):
            print("reading the existing superpoint graph file...")
            read_spg(room_spg_path)
        else:
            print("computing the superpoint graph...")
            #--- build the spg h5 file --
            features = np.hstack((desc, rgb))
            features[:,3] = 2. * features[:,3] #increase importance of verticality (heuristic)
            print("    minimal partition...")
            partition = libcp.cutpursuit(features.tolist(), graph_nn["source"].tolist(), graph_nn["target"].tolist(), reg_strength)
            components = np.array(partition[0], dtype = 'object')
            in_component = np.array(partition[1], dtype = 'uint32')
            del partition
            print("    computation of the SPG...")
            graph_sp = compute_sp_graph(xyz, d_se_max, in_component, components, room_labels, 13)
            write_spg(room_spg_path, graph_sp, components, in_component)
