import os

class ConfigParser():
    """Parser for vnmrjpy config file
    
    Config file is currently ./config/conf.
    Key-value pairs are simply given by key=value,
    no other structure is present yet.

    Methods:
        __init__(configfile)
        parse()
            return {key : val} dictionary

    """
    def __init__(self,configfile=None):

        default_config_path = os.path.dirname(os.path.dirname(\
                os.path.abspath(__file__)))+'/config'

        if configfile == None:
            # search for config file
            for loc in [os.curdir,\
                    os.path.expanduser("~"),\
                    default_config_path]:
                try:
                    with open(os.path.join(loc,'vnmrjpy.conf')) as conf:
                        self.configfile = os.path.join(loc,'vnmrjpy.conf')
                except:
                    pass
        else:
            self.configfile = configfile
        if self.configfile == None:
            raise(Exception('ConfigParser could not find vnmrjpy.conf'))


    def parse(self):
        """Return dictionary of config parameters"""
        conf_dict = {}
        with open(self.configfile,'r') as opencfg:
            for line in opencfg:
                line = line.rsplit('\n')[0]
                #exclude comments
                if '#' in line:
                    continue
                # key-value pairs:
                if '=' in line:
                    key = line.rsplit('=')[0]
                    val = line.rsplit('=')[1]
                    if '[' and ']' in val:
                        val_list = []
                        for item in val[1:-1].rsplit(','):
                            try:
                                item = int(item)
                            except:
                                try:
                                    item = float(item)
                                except:
                                    item = str(item)
                            val_list.append(item)
                        val = val_list
                    # check if integer
                    elif val.isdigit():
                        val = int(line.rsplit('=')[1])
                    else:
                        try:
                            val = float(val)
                        except:
                            pass
                        if val in ['False','false','FALSE']:
                            val = False
                        elif val in ['True','true','TRUE']:
                            val = True
                        else:
                            pass
                                
                    conf_dict[key] = val
        return conf_dict


if __name__ == '__main__':
    confparser = ConfigParser()
    confparser.parse()


