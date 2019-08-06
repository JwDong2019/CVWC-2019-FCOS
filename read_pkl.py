import os
import json
import pickle


def convert(predicted_file_path, json_file):



    with open(predicted_file_path, 'rb') as f:
        json_list = []
        data = pickle.load(f)
        #print(data[1651][0]['id'])
        i = 0
        for pre_data in data[0:1651]:
            #print(len(pre_data[0]))
            for j in range(len(pre_data[0])):

                xmin = pre_data[0][j][0]
                ymin = pre_data[0][j][1]
                xmax = pre_data[0][j][2]
                ymax = pre_data[0][j][3]

                image_id = int(data[1651][i]['id'])
                category_id = int(1)
                width = float(xmax)-float(xmin)
                height = float(ymax)-float(ymin)
                bbox = [float(xmin),float(ymin),width,height]
                score = float(pre_data[0][j][4])
                json_dict = {"image_id": image_id, "category_id": category_id, "bbox": bbox, "score": score}
                json_list.append(json_dict)
            i = i+1

        json_fp = open(json_file, 'w')
        json_str = json.dumps(json_list)
        json_fp.write(json_str)
        json_fp.close()



if __name__ == '__main__':
    predicted_file_path = "/home/djw/PycharmProjects/CVWC-2019-FCOS/tools/work_dir/3/output_e2.pkl"
    json_file = '/home/djw/Downloads/voc2coco/fcosv4_voc2coco_tiger.json'

    convert(predicted_file_path = predicted_file_path,json_file=json_file)