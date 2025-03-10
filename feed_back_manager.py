# Still thinking wheather to use joblib or json -sqlite3 or csv/ txt for storing the feedbacks.
#in streamlit it should

from IPython.display import Audio, display
DEF_REQ = 50 # Default Request interval for feedbacks is 50 predictions.

'''
def req_feedback(input_data, my_pred):
    global threshold
    my_current_score = 0
    "This method requsts feedback about its prediction by  providing the audio to user."
    display(Audio(input_data, rate = 22050))
    print(f"I predicted it as: {my_pred}")
    is_correct = input("Am I correct Y?N: ").strip().lower()
    if is_correct == 'y':
        threshold *= 2
        return False
    elif is_correct == 'n':
        user_label = input("please provide me correct label then..:").strip().lower()
        feedback = input_data, user_label, my_pred
        threshold //= 2
        return feedback
    else:
        print("Not a valid feedback! Aborting....feedback.")'''

def save_feedback(path, label):
    #('table', 'path_and_labels', 'path_and_labels', 2, 'CREATE TABLE path_and_labels (path, labels)')
    import sqlite3
    con = sqlite3.connect('/home/badri/mine/ser/gnd/capstone_project/Back_end/back_end/data/correct_labels.db')
    cur = con.cursor()
    cur.execute("INSERT INTO path_and_labels values(?,?)",(path, label))
    con.commit()
    con.close()
    print("succesfully taken feedback")


        

        
        
    
