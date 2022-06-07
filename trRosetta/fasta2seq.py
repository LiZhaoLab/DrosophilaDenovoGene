
import os,sys,gzip

def fasta2seq(fasta,progress_bar = 'off'):
    if os.path.isfile(fasta):
        if fasta.endswith(".gz"):
            p = gzip.open(fasta,'rt')
        else:
            p = open(fasta,'r')
        lines = p.readlines()
        p.close()
        sequence = {}
#        plines = tqdm(lines,ascii=True,unit='B',unit_scale=True,unit_divisor=1024)
#        plines.set_description("ReadFas")
#        for line in plines:
        nlines = len(lines)
        nperbar = max(nlines/1000,1)
        m = 0
        for i in range(nlines):
            if lines[i].startswith('>'):
                key = lines[i][1:].strip()
                sequence[key]  = []
                m += 1
            else:
                if m == 0:
                    if lines[i].strip():
                        break
                    else:
                        continue
                else:
                    sequence[key].append(lines[i].strip().upper())
            if progress_bar == 'on':
                if i%nperbar==0 or i==nlines-1:
                    percent = (i/(nlines-1))*100
                    sys.stdout.write("\rReadFas... %5.1f%%\r" % percent)
            i += 1
        if progress_bar == 'on':
            sys.stdout.write('\n')

        for key in sequence:
            sequence[key] = ''.join(sequence[key])

        return sequence

    else:
        if progress_bar == 'on':
            sys.stdout.write(">Error: %s not exist\n"%fasta)
        return {}

def fasta2phylip(fasta,name=''):
    sequence = fasta2seq(fasta)
    phylines = dict([[key,'%-10s'%"%s%s"%(name,key)] for key in sequence])

    for key in sequence:
        align_length = len(sequence[key])
        for j in range(len(sequence[key])):
            phylines[key] += sequence[key][j]
#            if (j+1)%10 == 0:
#                phylines[i] += ' '
        phylines[key] += '\n'

    fphylip = '%s.phylip'%os.path.basename(fasta)
    with open(fphylip,'w') as p:
        p.write("%10d %10d\n"%(len(phylines),align_length))
        for key in phylines:
            p.write(phylines[key])

def phylip2seq(phylip,name=''):
    if not os.path.isfile(phylip):
        print(">Error: phylip file %s not exist!"%phylip)
        sys.exit(1)

    sequence = dict()
    with open(phylip,'r') as p:
        lines = p.readlines()

        try:
            numseq,seqnum = [int(s) for s in lines[0].split()]
        except ValueError:
            print(">Error: phylip file %s is not formatted!"%phylip)
            sys.exit(1)

        actual_numseq = 0
        for line in lines[1:]:
            keyi,seqi = line.split()
            sequence[keyi] = seqi
            actual_numseq += 1

        if actual_numseq!=numseq:
            print(">Warn: no. of sequences declared is not consistent!")
        
    return sequence
        
#def translate(seq):
#    import translate_cds
#    seq_aa = {}
#    for key in seq:
#        seq_nt = seq[key]
#        seq_aa[key] = translate_cds.translate(seq_nt)
#    return seq_aa
    
def seq2fasta(seq,fasta,linebreak=False,compress=False):
    if compress:
        f = gzip.open("%s.gz"%fasta,'wb')
    else:
        f = open(fasta,'w')

    ## line break for txt and gz format ##
    lb = '\n'
    lb_byte = lb.encode('utf-8')

    for key in seq:
        seqi = seq[key]

        header = '>%s\n'%key   
        if compress:
            header = header.encode('utf-8')
        f.write(header)
        n = 1

        if linebreak:
            for s in seqi:
                if compress:
                    s = s.encode('utf-8')
                f.write(s)
                    
                if n%80 == 0:
                    if compress:
                        lb = lb_byte
                    f.write(lb)
                n += 1
        else:
            if compress:
                seqi = seqi.encode('utf-8')
            f.write(seqi)

        if compress:
            lb = lb_byte
        f.write(lb)

def seq2phylip(seq,outfile,name=''):
    phylines = dict([[key,'%-10s'%"%s%s"%(name,key)] for key in seq])

    for key in seq:
        key = key
        break
    align_length = len(seq[key])
    new_length   = 0
    for j in range(align_length):
        gaps = [seq[key][j] for key in seq]
        if not all([s=="-" for s in gaps]):
            new_length += 1
            for key in seq:
                phylines[key] += seq[key][j]
    for key in seq:
        phylines[key] += '\n'

    with open(outfile,'w') as p:
        p.write("%10d %10d\n"%(len(phylines),new_length))
        for key in phylines:
            p.write(phylines[key])


if __name__ == '__main__':
    fasta = '/ru-auth/local/home/jpeng/scratch/DrosophilaProteins/LiftOver/KaKsTest/dmel_group/fasta/FBgn0003495.fa'
    sequence = fasta2seq(fasta)
    #print(sequence)

    #print('This is OK?')
    seq_aa = translate(sequence)
    #print(seq_aa)

    from pdb2seq import printscreen
    printscreen(seq_aa)
