import os
import numpy as np
import vnmrjpy as vj

class FidReader():
    """Handles raw data from Varian spectrometer

    .fid File structure as per Vnmrj manual:
    ===================================
    struct datafilehead
    Used at the beginning of each data file (fid's, spectra, 2D) 

       int     nblocks;      /* number of blocks in file
       int     ntraces;      /* number of traces per block    
       int     np;           /* number of elements per trace    
       int     ebytes;       /* number of bytes per element            
       int     tbytes;       /* number of bytes per trace        
       int     bbytes;       /* number of bytes per block        
       short   vers_id;      /* software version and file_id status bits
       short   status;       /* status of whole file    
       int       nbheaders;     /* number of block headers            

    struct datablockhead
     Each file block contains the following header 

       short   scale;    /* scaling factor        
       short   status;    /* status of data in block      
       short   index;    /* block index              
       short   mode;    /* mode of data in block     
       int       ctcount;    /* ct value for FID        
       float   lpval;    /* F2 left phase in phasefile    
       float   rpval;    /* F2 right phase in phasefile
       float   lvl;        /* F2 level drift correction       
       float   tlt;        /* F2 tilt drift correction      
    """

    def __init__(self,fid,procpar=None):
        """Decodes .fid file header"""

        if os.path.isdir(fid):
            procpar = str(fid)+'/procpar'
            self.procpar = procpar
            fid = str(fid)+'/fid'
        else:            
            self.procpar = procpar

        def _decode_header():

            hkey_list = ['nblocks','ntraces','np','ebytes','tbytes','bbytes',\
                     'vers_id','status','nbheaders']
            hval_list = []
            h_dict = {}
            if len(self.bheader) != 32:
                print('incorrect fid header data: not 32 bytes')
                return -1
            else: 
                hval_list.append(int.from_bytes(self.bheader[0:4],byteorder='big'))
                hval_list.append(int.from_bytes(self.bheader[4:8],byteorder='big'))
                hval_list.append(int.from_bytes(self.bheader[8:12],byteorder='big'))
                hval_list.append(int.from_bytes(self.bheader[12:16],byteorder='big'))
                hval_list.append(int.from_bytes(self.bheader[16:20],byteorder='big'))
                hval_list.append(int.from_bytes(self.bheader[20:24],byteorder='big'))
                hval_list.append(int.from_bytes(self.bheader[24:26],byteorder='big'))
                hval_list.append(int.from_bytes(self.bheader[26:28],byteorder='big'))
                hval_list.append(int.from_bytes(self.bheader[28:],byteorder='big'))

                for num, i in enumerate(hkey_list):
                    h_dict[i] = hval_list[num]
            
                self.header_dict = h_dict

                def decode_status(): #TODO
                    pass

        # data should be 32 bytes
        with open(fid,'rb') as openFid:
            binary_data = openFid.read() 
            self.bheader = bytearray(binary_data[:32])
            self.bdata = binary_data[32:]

        _decode_header()

    def read(self):
        """Reads raw data, arranges into numpy array

        Return:
            data -- numpy.ndarray([blocks, data])
            header -- dictionary of fid header
    
        """

        def chunks(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        chunk_size = int(self.header_dict['bbytes'])
        block_list = list(chunks(self.bdata,chunk_size))

        if self.header_dict['bbytes'] == len(block_list[0]):
            #print('blocksize check OK')
            pass

        def decode_blockhead(blockheader):
            bh_dict = {}
            bhk_list = ['scale','status','index','mode','ctcount','lpval'\
                        'rpval','lvl','tlt']
            bhv_list = []
            bhv_list.append(int.from_bytes(blockheader[0:2],byteorder='big'))
            bhv_list.append(int.from_bytes(blockheader[2:4],byteorder='big'))
            bhv_list.append(int.from_bytes(blockheader[4:6],byteorder='big'))
            bhv_list.append(int.from_bytes(blockheader[6:8],byteorder='big'))
            bhv_list.append(int.from_bytes(blockheader[8:12],byteorder='big'))
            bhv_list.append(int.from_bytes(blockheader[12:16],byteorder='big'))
            bhv_list.append(int.from_bytes(blockheader[16:20],byteorder='big'))
            bhv_list.append(int.from_bytes(blockheader[20:24],byteorder='big'))
            bhv_list.append(int.from_bytes(blockheader[24:28],byteorder='big'))

            for num, i in enumerate(bhk_list):
                bh_dict[i] = bhv_list[num]        
            self.blockhead_dict = bh_dict

        #------------main iteration through bytearray -----------------

        dim = (int(self.header_dict['nblocks']),\
                int((self.header_dict['ntraces'])*int(self.header_dict['np'])))
        DATA = np.empty(dim)
        if self.header_dict['ebytes'] == 4:
            dt = '>f'
        if self.header_dict['ebytes'] == 2:
            dt = '>i2'
        for k,block in enumerate(block_list): # for each block
            block_header = block[:28] # separate block header
            bh = decode_blockhead(block_header)
            self.print_blockheader()
            block_data = block[28:]
            DATA[k,:] = np.frombuffer(bytearray(block_data),dt)

        #self.blockhead_dict = decode_blockhead(block_header)
        return DATA, self.header_dict

    def print_header(self):
        """Prints fid header to stdout"""

        print('---------------Fid header---------------------')
        for i in sorted(self.header_dict.keys()):
            print(str(i)+' = '+str(self.header_dict[i]))

    # after read method
    def print_blockheader(self):
        
        print('---------------block header-------------------')
        for i in sorted(self.blockhead_dict.keys()):
            print(str(i)+' = '+str(self.blockhead_dict[i]))
        print('----------------------------------------------')


    
    def make_image(self):
        """Fully convert fid to 4D image as numpy array.

        This is a convenience function for testing purposes, using this
        is not advised for serious work. Relies on vj.recon mostly.

        Return:
            image (np.ndarray) -- summed image
            affine -- affine for nibabel

        """
        def _ssos(data5D):
            """Combine receivers with square root of sum of squares method"""
            ssos = np.sqrt(np.mean(np.square(np.absolute(data5D)),axis=0))            
            return ssos

        data, header = self.read()
        kspace = vj.recon.KspaceMaker(data, header, self.procpar).make()
        imgspace = vj.recon.ImageSpaceMaker(kspace, self.procpar).make()
        image = _ssos(imgspace)
        image = vj.util.to_scanner_space(image, self.procpar)
        #nwr = vj.io.NiftiWriter(image, self.procpar)
        affine = vj.util.make_scanner_affine(self.procpar)
        return image, affine
        







