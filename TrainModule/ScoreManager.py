


'''
Caculating Accuracy
'''
class ScoreManager():
    def __init__(self) -> None:
        pass
    
        
    def hit_rate(self, y_true, y_pred, k, sequence = True):
        '''
        Recording hit if target is in top-k items
        k : number to choose top # items
        
        W/O Sequence
        y_true (Batch,): label of output - index type    
        y_pred (Batch,Items): prediction output
        
        W Sequence
        y_true (Batch, Seq)
        y_pred (Batch, Seq, Items)
        '''
        
        if not sequence:
            y_pred = y_pred.numpy()
            length = len(y_pred)
            
            hit = 0
            for i in range(length):
                indices = (-y_pred[i]).argsort()[:k]
                if y_true[i] in indices:
                    hit += 1
                    
            hit_rate = hit / length
            
            return hit_rate
        else:
            y_pred = y_pred.numpy()
            batch_size = len(y_pred)
            sequence_len = len(y_pred[0])
            
            hit = 0
            total_num = 0
            for i in range(batch_size):
                for j in range(sequence_len):
                    true_idx = y_true[i][j]
                    
                    if true_idx != 0:
                        indices = (-y_pred[i][j]).argsort()[:k]
                        if true_idx in indices:
                            hit += 1
                        
                        total_num += 1
                        
            hit_rate = hit / total_num
            
            return hit_rate

        
        