import json

'''
    a elite student can be defined the following two points:
    1. elite_log_num >= 900
        if a student's log_num exceeds 500,we can say the student is elite.
    2. test_accuracy >= 0.7
        if a student's test_accuracy exceeds 0.5,we can say the student is elite.
'''
elite_log_num = 900
test_accuracy = 0.7

def find_elite_students():
    with open('../data/junyi/log_data_all_raw.json', encoding='utf8') as i_f:
        stus = json.load(i_f)
    elite_log_data_all = []
    for stu in stus:
        log_num = stu['log_num']
        correct = 0
        logs = []
        for log in stu['logs']:
            correct+=log['score']
            logs.append(log)
        if log_num >= elite_log_num or correct/log_num >= test_accuracy:
            user_id = stu['user_id']
            stu_set = {'user_id' : user_id}
            stu_set['log_num'] = log_num
            stu_set['logs'] = logs
            elite_log_data_all.append(stu_set)
    # print("all_studens_count = ", len(stus))
    # print("excellent_students_count = ", len(elite_log_data_all))
    with open('../data/junyi/log_data.json', 'w', encoding='utf8') as output_file:
        json.dump(elite_log_data_all, output_file, indent=4, ensure_ascii=False)    # 直接用test_set作为val_set

if __name__ == '__main__':
    find_elite_students()
        