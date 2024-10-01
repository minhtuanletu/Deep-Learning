import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class Compute_Metrics:
    def __init__(self):
        pass
    
    def process_output(self, output):
        output = torch.argmax(output, dim=-1)
        output = output.tolist()
        return output
        
    def compute_accuracy(self, y, y_pred):
        accuracy = accuracy_score(y_true=y, y_pred=y_pred)
        return accuracy
    
    def compute_recall(self, y, y_pred):
        recall = recall_score(y_true=y, y_pred=y_pred, average='micro')
        return recall
    
    def compute_precision(self, y, y_pred):
        precision = precision_score(y_true=y, y_pred=y_pred, average='micro')
        return precision
    
    def compute_f1(self, y, y_pred):
        f1 = f1_score(y_true=y, y_pred=y_pred, average='micro')
        return f1
    
    def compute(self, y, y_pred):
        y = self.process_output(y)
        y_pred = self.process_output(y_pred)
        accuracy = self.compute_accuracy(y, y_pred)
        precision = self.compute_precision(y, y_pred)
        recall = self.compute_recall(y, y_pred)
        f1 = self.compute_f1(y, y_pred)
        print(accuracy, precision, recall, f1)
        return accuracy, precision, recall, f1