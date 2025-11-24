import numpy as np

def scale_up_runtime(mnk_data, hw_config):
    """
    Calculate the runtime for a matrix multiplication operation on a systolic array.
    
    Args:
        mnk_data: List containing M, N, K dimensions of the matrix multiplication
        hw_config: Dictionary containing hardware configuration parameters
        
    Returns:
        Runtime in cycles
    """
    # Convert string values to integers if needed
    m = int(mnk_data[0])
    n = int(mnk_data[1])
    k = int(mnk_data[2])
    
    # Check the dataflow
    if hw_config['dataflow'] == 'OS':
        SR = m
        SC = n
        T = k
    elif hw_config['dataflow'] == 'WS':
        SR = k
        SC = n
        T = m
    elif hw_config['dataflow'] == 'IS':
        SR = k
        SC = m
        T = n
    
    # Get systolic array dimensions
    R = hw_config['sa_row']
    C = hw_config['sa_col']
    
    # Basic runtime calculation - implement your specific algorithm here
    print(f"Calculating runtime for dimensions M={m}, N={n}, K={k} on {R}x{C} systolic array")
    
    # Placeholder - replace with actual calculation
    runtime = (2*R + C + T - 2) * int(np.ceil(SR/R)) * int(np.ceil(SC/C))
    if hw_config['dataflow'] == 'OS':
        runtime = (R + C + T - 2) * int(np.ceil(SR/R)) * int(np.ceil(SC/C))
        
    return runtime

def scale_up_buf_access(mnk_data, hw_config):
    """
    Calculate buffer access counts for input, weight, and output buffers.
    
    Args:
        mnk_data: List containing M, N, K dimensions of the matrix multiplication
        hw_config: Dictionary containing hardware configuration parameters
        
    Returns:
        Tuple of (input_buffer_accesses, weight_buffer_accesses, output_buffer_accesses)
    """
    # Convert string values to integers if needed
    m = int(mnk_data[0])
    n = int(mnk_data[1])
    k = int(mnk_data[2])
    
    I_ROW = m
    I_COL = k
    W_ROW = k
    W_COL = n
    O_ROW = m
    O_COL = n
    # Get systolic array dimensions
    SA_ROW = hw_config['sa_row']
    SA_COL = hw_config['sa_col']
    
    print(f"Calculating buffer accesses for dimensions M={m}, N={n}, K={k}")
    
    if hw_config['dataflow'] == 'OS':
        input_buf_access = int(np.ceil(W_COL/SA_COL)) * I_ROW * I_COL
        weight_buf_access = int(np.ceil(I_ROW/SA_ROW)) * W_ROW * W_COL
        output_buf_access = O_ROW * O_COL
    elif hw_config['dataflow'] == 'WS':
        input_buf_access = int(np.ceil(W_COL/SA_COL)) * I_ROW * I_COL
        weight_buf_access = W_ROW * W_COL
        output_buf_access = int(np.ceil(W_ROW/SA_ROW)) * O_ROW * O_COL
    elif hw_config['dataflow'] == 'IS':
        input_buf_access = I_ROW * I_COL
        # weight_buf_access = int(np.ceil(I_COL/SA_COL)) * W_ROW * W_COL ###
        # output_buf_access = int(np.ceil(I_ROW/SA_ROW)) * O_ROW * O_COL
        weight_buf_access = int(np.ceil(I_ROW/SA_ROW)) * W_ROW * W_COL ###
        output_buf_access = int(np.ceil(I_COL/SA_COL)) * O_ROW * O_COL
    
    return (input_buf_access, weight_buf_access, output_buf_access)

def scale_up_off_access(mnk_data, hw_config):
    """
    Calculate off-chip memory access counts for input, weight, and output data.
    
    Args:
        mnk_data: List containing M, N, K dimensions of the matrix multiplication
        hw_config: Dictionary containing hardware configuration parameters
        
    Returns:
        Tuple of (input_offchip_accesses, weight_offchip_accesses, output_offchip_accesses)
    """
    # Convert string values to integers if needed
    m = int(mnk_data[0])
    n = int(mnk_data[1])
    k = int(mnk_data[2])
    
    I_ROW = m
    I_COL = k
    W_ROW = k
    W_COL = n
    O_ROW = m
    O_COL = n
    
    # Get buffer sizes and other relevant configuration
    input_buf_size = hw_config['input_buf_size']
    weight_buf_size = hw_config['weight_buf_size']
    output_buf_size = hw_config['output_buf_size']
    dataflow = hw_config['dataflow']
    
    # Get systolic array dimensions
    SA_ROW = hw_config['sa_row']
    SA_COL = hw_config['sa_col']
    
    print(f"Calculating off-chip accesses for dimensions M={m}, N={n}, K={k} with {dataflow} dataflow")
    
    if dataflow == 'OS':
        # input_off_access = I_ROW * I_COL  # Input matrix reads
        if int(np.floor(input_buf_size) * 1/2) >= I_ROW * I_COL:
            input_off_access = I_ROW * I_COL
        else:
            input_off_access = int(np.ceil(W_COL/SA_COL)) * I_ROW * I_COL
        # weight_off_access = W_ROW * W_COL  # Weight matrix reads
        if int(np.floor(weight_buf_size) * 1/2) >= W_ROW * W_COL:
            weight_off_access = W_ROW * W_COL
        else:
            weight_off_access = int(np.ceil(I_ROW/SA_ROW)) * W_ROW * W_COL
        output_off_access = O_ROW * O_COL  # Output matrix writes
    
    elif dataflow == 'WS':
        if int(np.floor(input_buf_size) * 1/2) >= I_ROW * I_COL:
            input_off_access = I_ROW * I_COL
        else:
            input_off_access = int(np.ceil(W_COL/SA_COL)) * I_ROW * I_COL
        weight_off_access = W_ROW * W_COL
        output_buf_access = int(np.ceil(W_ROW/SA_ROW)) * O_ROW * O_COL ###
        output_off_access = output_buf_access
    
    elif dataflow == 'IS':
        input_off_access = I_ROW * I_COL
        # weight_off_access = W_ROW * W_COL
        if int(np.floor(weight_buf_size) * 1/2) >= W_ROW * W_COL:
            weight_off_access = W_ROW * W_COL
        else:
            weight_off_access = int(np.ceil(I_ROW/SA_ROW)) * W_ROW * W_COL
        output_buf_access = int(np.ceil(I_COL/SA_COL)) * O_ROW * O_COL ###
        output_off_access = output_buf_access
    
    return (input_off_access, weight_off_access, output_off_access)
