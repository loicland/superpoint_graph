import numpy as np
import open3d
import os
from bs4 import BeautifulSoup

def display_cloud(clouds,las_header=None,labels = [],display = False,size =1000000): 
    """
    Takes a list of pcl or open3d point clouds together with the input file's header and
    uses Potree converter to generate the viewable assests. You can view the cloud at the following URL
    https://jupyterhub.helix.re/user/nicolas/proxy/3221/cloud_viewer.html"
    """
    PAGE_TEMPLATE = '/opt/potree/dev/workspaces/PotreeConverter/master/PotreeConverter/resources/page_template/'
    os.popen('ps -ef | grep "http.server" | awk \'{print $2}\' | xargs kill -9')
    potree_converter = '/usr/local/bin/PotreeConverter'

    if not os.path.exists('viewer'):
        os.popen('mkdir viewer').readlines()
    else:
        os.popen('rm -rf viewer').readlines()
        os.popen('mkdir viewer').readlines()
    
    #load pt cloud
    os.popen(potree_converter+' p.ply -o viewer --page-template %s -p cloud_viewer --overwrite' % PAGE_TEMPLATE).readlines()
    os.popen('rm -rf viewer/pointclouds/cloud_viewer').readlines()
    temp_file = "temp"
    try:
        os.remove(temp_file+'.las')
        os.remove(temp_file+'.pcd')
        os.remove(temp_file+'.ply')
    except: pass
    
    with open("viewer/cloud_viewer.html") as fp:
        main_html = BeautifulSoup(fp, "lxml")
    
    string_script= 'window.viewer = new Potree.Viewer(document.getElementById("potree_render_area"));viewer.setEDLEnabled(true);viewer.setFOV(60);viewer.setPointBudget(1*1000*1000);document.title = "";viewer.setEDLEnabled(false);viewer.setBackground("white");viewer.setDescription(``);viewer.loadSettingsFromURL();viewer.loadGUI(() => {viewer.setLanguage("en");$("#menu_appearance").next().show();$("#menu_tools").next().show();$("#menu_scene").next().show();viewer.toggleSidebar();});'
    
    soup = BeautifulSoup('<script></script>', "lxml")
    soup.find('script').string = string_script
    main_html.find_all('script')[-1].replace_with(soup.head.script)
    
    colors = np.random.rand(len(clouds),3)

    # decimates and saves to file
    for i in range(len(clouds)):
        cloud = clouds[i]
        cloud_arr = np.asarray(cloud.points)
        if i < len(labels):
            label = str(labels[i])
        else:
            label = str(i)

        if size < cloud_arr.shape[0]:
            idx = np.random.randint(cloud_arr.shape[0], size=int(size))
            cloud = open3d.select_down_sample(cloud,idx)
            
        open3d.write_point_cloud(temp_file+'.ply',cloud)


        #Point Cloud form ply to Potree
        os.popen(potree_converter + ' '+ temp_file+'.ply -o viewer/pointclouds/'+str(i) +' --incremental').readlines()
        add_pt_cloud = BeautifulSoup('<script>Potree.loadPointCloud("pointclouds/'+str(i) +'/cloud.js", "'+label+'", e => { let pointcloud = e.pointcloud;let material = pointcloud.material;viewer.scene.addPointCloud(pointcloud);material.pointColorType = Potree.PointColorType.COLOR;material.color = new THREE.Color().setRGB('+str(colors[i][0])+','+str(colors[i][1])+','+str(colors[i][2])+');material.size = 1;material.pointSizeType = Potree.PointSizeType.FIXED;material.shape = Potree.PointShape.SQUARE;viewer.fitToScreen();});</script>', "lxml")
        main_html.find_all('script')[-1].append(add_pt_cloud.script.string)
        os.remove(temp_file+'.ply')
    
    with open("viewer/cloud_viewer.html", "w") as file:
        file.write(str(main_html))
        
    #launch server on port 9000
    print("launching server")
    os.popen('cd viewer && python -m http.server 3221')
    print("server launched, go to https://jupyterhub.helix.re/user/nicolas/proxy/3221/cloud_viewer.html")