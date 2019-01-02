class ProcparReader():
    """Parses vnmrj procpar file."""

    def __init__(self, procpar_file):
        """Init procpar file reader
        
        Arguments:
        ==========
        procpar_file -- path/to/procpar
        """
        self.ppfile = str(procpar_file)

    def read(self):
        """Return a dictionary of procpar file parameters"""

        with open(self.ppfile,'r') as openpp:
            
            name = []
            value = []
            subtype = []
            basictype = []
            maxvalue = []
            minvalue = []
            # possibilities (as of manual): 0 :'line1', 1: 'line2', 2: 'last'
            line_id = 0 
        
            current_name = 0
    
            for line_num, line in enumerate(openpp.readlines()):

                if line_id == 0:  # parameter name line
                    
                    fields = line.rsplit('\n')[0].rsplit(' ')
                    current_name = fields[0]
                    name.append(fields[0])
                    subtype.append(int(fields[1]))
                    basictype.append(int(fields[2]))

                    # can be 0 (undefined), 1 (real), 2 (string)
                    current_basictype = int(fields[2])  
                    val_list = []  # if parameter value is a multitude of strings
                    line_id = 1
                                    
                elif line_id == 1:  # parameter value line

                    fields = line.rsplit('\n')[0].rsplit(' ')

                    # values are real, and all are on this line
                    if current_basictype == 1:  
                        val = [fields[i] for i in range(1,len(fields)-1)]
                        if len(val) == 1:  # don't make list if there is only one value
                            val = val[0]
                        value.append(val)
                        line_id = 2
                    # values are strings
                    elif current_basictype == 2 and int(fields[0]) == 1:
                        
                        value.append(str(fields[1].rsplit('"')[1]))
                        line_id = 2
                    # multiple string values
                    elif current_basictype == 2 and int(fields[0]) > 1:
                        
                        val_list.append(fields[1])
                        remaining_values = int(fields[0])-1
                        line_id = 3
                    
                elif line_id == 2:

                    line_id = 0
            
                elif line_id == 3:  # if values are on multiple lines
                    if remaining_values > 1:  # count backwards from remaining values
                        val_list.append(fields[0])
                        remaining_values = remaining_values - 1
                    elif remaining_values == 1:
                        val_list.append(fields[0])
                        remaining_values = remaining_values - 1
                        value.append(val_list)
                        line_id = 2
                    else:
                        print('something is off')

                else:
                    print('incorrect line_id')


        return dict(zip(name, value))
    
