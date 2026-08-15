import numpy as np

# 0 - free space
# int > 0 - path
# -1 - obstacle
# -2 - contact source
# -3 - contact end

class Wave_Route:
    
    def __init__(self, a, start_z, start_y, start_x, end_z, end_y, end_x):
        self.a = a
        self.start_y = start_y
        self.start_x = start_x
        self.start_z = start_z
        self.end_y = end_y
        self.end_x = end_x
        self.end_z = end_z
        a[self.start_y][self.start_x][self.start_z] = -2
        a[self.end_y][self.end_x][self.end_z] = -3
        
    def validate_current_state(self, array, z, y, x, index):
        
        row_len, column_len, height_len = array.shape
        new_positions_found = 0
        # print(height_len)
        # print(z)
        # outer IF elements validate position of the current state being -1 to every border
        # to prevent going over them with the next step
        
        if y > 0:
            if array[y - 1, x, z] == 0:
                array[y - 1, x, z] = index
                new_positions_found += 1           
            if array[y - 1, x, z] == -3: # exit condition (contact end) check
                return array, True, new_positions_found
            
        if y < row_len - 1:
            if array[y + 1, x, z] == 0:
                array[y + 1, x, z] = index
                new_positions_found += 1                        
            if array[y + 1, x, z] == -3: # exit condition (contact end) check
                    return array, True, new_positions_found
        
        
        if x > 0:
            if array[y, x - 1, z] == 0:
                array[y, x - 1, z] = index   
                new_positions_found += 1                  
            if array[y, x - 1, z] == -3: # exit condition (contact end) check
                    return array, True, new_positions_found        
    
        if x < column_len - 1:
            if array[y, x + 1, z] == 0:
                array[y, x + 1, z] = index 
                new_positions_found += 1           
            if array[y, x + 1, z] == -3: # exit condition (contact end) check
                    return array, True, new_positions_found


        if z > 0:
            if array[y, x, z - 1] == 0:
                array[y, x, z - 1] = index
                new_positions_found += 1      
            if array[y, x, z - 1] == -3: # exit condition (contact end) check
                return array, True, new_positions_found
            
        if z < height_len - 1:
            if array[y, x, z + 1] == 0:
                array[y, x, z + 1] = index
                new_positions_found += 1                 
            if array[y, x, z + 1] == -3: # exit condition (contact end) check
                return array, True, new_positions_found

        
        return array, False, new_positions_found

        
    def wave_generator(self, array, z, y, x):
        
        row_len, column_len, height_len = array.shape
        prev_array = np.copy(array)
        total_positions = 0
        wave_number = 1
        
        array, exit, new_positions_found = self.validate_current_state(array, z, y, x, wave_number)
        
        if new_positions_found == 0:
            return array, wave_number - 1, False
        
        wave_number += 1
        exit_while = False
        while exit_while == False:
            
            prev_array = np.copy(array)
            # print(array.shape,row_len, column_len, height_len)
            for k in range (height_len):
                for i in range(row_len):
                    for j in range(column_len):
                        if array[i, j, k] == wave_number - 1:

                            array, exit, new_positions_found = self.validate_current_state(array, k, i, j, wave_number)
                            total_positions += new_positions_found

                            if exit == True:
                                exit_while = True
                                array = prev_array
                                break
                            
            if total_positions == 0:
                return  array, wave_number - 1, False
                        
            total_positions = 0                         
            wave_number += 1         

        return array, wave_number - 1, True


    def path_backtracking(self, array, z, y, x, wave_number):
        
        path = np.zeros((0,3), dtype=int) 
        path = np.append(path, [[self.end_y, self.end_x, self.end_z]], axis=0)

        row_len, column_len, height_len = array.shape

        for i in range(wave_number, -1, -1):

            # block below checks the positions for the next step of wave algorithm
            # 0 x 0
            # x c x
            # 0 x 0
            # x - checked positions
            # c - current position
            
            if y > 0:
                if array[y - 1, x, z] == i:         
                    path = np.append(path, [[y - 1, x, z]], axis=0)
                    y = y - 1  
            if y < row_len - 1:
                if array[y + 1, x, z] == i:
                    path = np.append(path, [[y + 1, x, z]], axis=0)
                    y = y + 1          
                    
            if x > 0:
                if array[y, x - 1, z] == i:
                    path = np.append(path, [[y, x - 1, z]], axis=0)
                    x = x - 1          
            if x < column_len - 1:
                if array[y, x + 1, z] == i:
                    path = np.append(path, [[y, x + 1, z]], axis=0)                        
                    x = x + 1
                    
            if z > 0:
                if array[y, x, z - 1] == i:
                    path = np.append(path, [[y, x, z - 1]], axis=0)
                    z = z - 1     
            if z < height_len - 1:
                if array[y, x, z + 1] == i:
                    path = np.append(path, [[y, x, z + 1]], axis=0)                   
                    z = z + 1
                        
        path = np.append(path, [[self.start_y, self.start_x, self.start_z]], axis=0)

        return(path)


    def final_route(self, array, path):
        for i in path :
            array[i[0], i[1], i[2]] = 1
        return array

    def output(self):
        row_len, column_len, height_len = self.a.shape
        a_zeros = np.zeros((row_len, column_len, height_len), dtype=float)
        
        array, wave_number, success = self.wave_generator(self.a, self.start_z, self.start_y, self.start_x)
        
        if success == False:
            return a_zeros, wave_number
        
        path = self.path_backtracking(array, self.end_z, self.end_y, self.end_x, wave_number)
        array_output = self.final_route(a_zeros, path)
        
        return array_output, wave_number
