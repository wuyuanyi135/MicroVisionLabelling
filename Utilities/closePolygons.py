import sys
import csv
import json
import os

csv_file_path = sys.argv[1]
csv_output_path = os.path.join(os.path.dirname(csv_file_path), 'closed_annotations.csv')
with open(csv_file_path, 'r') as csvfile, open(csv_output_path, 'w', newline='') as csvwritefile:
    csv_reader = csv.reader(csvfile)
    csv_writer = csv.writer(csvwritefile)
    csv_writer.writerow(next(csv_reader))
    for row in csv_reader:
        obj = json.loads(row[5])
        if obj['name'] == 'polygon':
            x = obj['all_points_x']
            y = obj['all_points_y']

            if x[0] != x[-1] or y[0] != y[-1]:
                x.append(x[0])
                y.append(y[0])
                
                obj['all_points_x'] = x
                obj['all_points_y'] = y
        row[5] = json.dumps(obj)
        csv_writer.writerow(row)
            
