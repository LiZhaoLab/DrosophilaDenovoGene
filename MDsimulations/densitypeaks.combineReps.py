""" 
    Density peaks clustering using TMscore as metrics
"""

from multiprocessing import Pool
import os,sys,time,shutil,subprocess
import numpy as np
import pandas as pd

def get_genenames():
    genenames = dict()
    flist    = "genenames.txt"
    lines    = open(flist,"r")
    for line in lines:
        pref,_,name = line[:-1].split()
        genenames[pref] = name

    return genenames

def extractPDBFromXTC(xtc,pdb,tpr,t):
    min_t = 0.001
    tu  = 1000.0
    tb  = "%.3f"%(t*tu)
    te  = "%.3f"%(t*tu+min_t)
    cmd = "gmx trjconv"
    cmd = "echo 0 | %s -f %s -s %s -o %s -b %s -e %s"%(cmd,xtc,tpr,pdb,tb,te)
    try:
        subprocess.check_output(cmd,shell=True,stderr=subprocess.STDOUT)
        print(cmd)
        return True
    except subprocess.CalledProcessError:
        print(cmd)
        return False

def extractOneRep(pref,fxtc,nf,tpr,t1,t2,dt,fpdblist,repn,mode='a',ncpu=1):
    print("    Extracting trajectories From %s..."%fxtc)
    #print(nf,t1,t2,tu)
    pdb  = "/tmp/mdfolder/%s_md_%s_%d.pdb"%(pref,repn,nf-1)
    flag = extractPDBFromXTC(fxtc,pdb,tpr,t2) 
    if not flag:
        sys.exit("%s not complete"%fxtc)

    if ncpu > 1:
        pool = Pool(ncpu)
    
    ## append PDB names into fpdblist 
    f = open(fpdblist,mode=mode)

    for i in range(nf):
        t = t1 + i*dt
        pdb  = "/tmp/mdfolder/%s_md_%s_%d.pdb"%(pref,repn,i)
        f.write("%s\n"%pdb)
        if ncpu>1:
            pool.apply_async(extractPDBFromXTC,(fxtc,pdb,tpr,t,))
        else:
            extractPDBFromXTC(fxtc,pdb,tpr,t) 
    if ncpu>1:
        pool.close()
        pool.join()
    
    #close file
    f.close()

def extractMultipleReps(pref,numrep,nf,fpdblist,tpr,t1,t2,dt,mdfolder,ncpu=1):
    mode = "w"
    for i in range(1,numrep+1):
        repn = "rep%d"%i
        fxtc = "%s/%s_md_%s.fit.xtc"%(mdfolder,pref,repn)
        if repn!="rep1":
            mode = "a"
        extractOneRep(pref,fxtc,nf,tpr,t1,t2,dt,fpdblist,repn,mode=mode,ncpu=ncpu)


def tmalign(pdb1,pdb2,byresi=False):
    cmd = 'TMalign %s %s'%(pdb1,pdb2)
    if byresi:
        cmd = 'TMalign %s %s -byresi 1'%(pdb1,pdb2)
    ##TMalign gives better TMscore and is faster than TMscore
    result = subprocess.check_output(cmd,shell=True).decode(sys.stdout.encoding)
    iflag  = 0
    gdt = 0
    rmsd = 0
    for r in result.split("\n"):
        if r.startswith("GDT-TS-score"):
            gdt  = float(r[14:20])
        if r.startswith("TM-score"):
            if iflag==0:
                tm = float(r.split()[1])
            iflag += 1
        if r.startswith("Aligned length"):
            rmsd = float(r.split()[4][:-1])
    return gdt,tm,rmsd

def tmscore(pdb1,pdb2,idx,metric="TM"):
    cmd = 'TMalign %s %s'%(pdb1,pdb2)
    ##TMalign gives better TMscore and is faster than TMscore
    result = subprocess.check_output(cmd,shell=True).decode(sys.stdout.encoding)
    iflag  = 0
    #if metric == "TM":
    #    for r in result.split("\n"):
    #        if r.startswith("TM-score"):
    #            tm = float(r.split()[1])
    #            break
    #if metric == "GDT":
    #    for r in result.split("\n"):
    #        if r.startswith("GDT-TS-score"):
    #            tm = float(r[14:20])
    #            break
    #if metric == "RMSD":
    #    for r in result.split("\n"):
    #        if r.startswith("Aligned length"):
    #            tm = float(r.split()[4][:-1])
    #            break
    for r in result.split("\n"):
        if r.startswith("TM-score"):
            tm   = float(r.split()[1])
        if r.startswith("Aligned length"):
            rmsd = float(r.split()[4][:-1])
    return tm,rmsd,idx

def maxsub(pdb,pdblist,numrep,repn,nf,idx,metric="TM"):
    cmd = 'maxcluster64bit -e %s -l %s'%(pdb,pdblist)
    result = subprocess.check_output(cmd,shell=True).decode(sys.stdout.encoding)
    metric = "TM" if metric=="GDT" else metric
    tmscores = [0. for s in range(numrep*nf)]
    rmsds    = [0. for s in range(numrep*nf)]
    idx3 = 0
    for r in result.split("\n"):
        if "TM" in r and "RMSD" in r:
            #if metric=="RMSD":
            #    tm = float(r.split()[9][:-1])
            #else:
            #    tm = float(r.split()[11][3:-1])
            rmsd = float(r.split()[9][:-1])
            tm   = float(r.split()[11][3:-1])
            pdb2 = r.split()[5]
            idx_rep,idx2 = pdb2[:-4].split("_")[-2:]
            idx_rep = int(idx_rep[3:])
            idx2 = int(idx2)
            idx3 = (idx_rep-1)*nf + idx2
            #print(repn,idx_rep,idx,idx2,idx3,tm,rmsd)
            
            tmscores[idx3] = tm
            rmsds[idx3] = rmsd
            #print(r,tm,idx2)
            idx3 += 1
    #print(len(rmsds),rmsds)
    #print(repn,idx,tmscores)
    #print(repn,idx,rmsds)
    return [tmscores,rmsds,repn,idx]

def get_dmat(pref,numrep,fpdblist,tpr,t1,t2,dt,mdfoler,redo,ncpu=1,metric="TM",method="maxsub"):
    """
        method: maxsub or TMalign
    """

    min_t = 0.001
    tu    = min_t
    nf    = int((t2-t1)/dt+1)
    print(pref,numrep,nf,fpdblist)

    ftm   = "%s.tmscore.txt"%(pref)
    frmsd = "%s.rmsd.txt"%(pref)
    fmat  = "%s.matrix.txt"%(pref)

    if os.path.exists(fmat) and redo==False:
    #if redo==False:
        if metric=="TM" and os.path.exists(ftm):
            tmscores = np.loadtxt(ftm)
            dmat  = 1-tmscores
        elif metric=="RMSD" and os.path.exists(frmsd):
            rmsds = np.loadtxt(frmsd)
            dmat  = rmsds 

        if numrep*nf == len(dmat):
            return dmat
    
    print("    Extracting all trajectories from %s..."%pref)

    extractMultipleReps(pref,numrep,nf,fpdblist,tpr,t1,t2,dt,mdfolder,ncpu=ncpu)

    if ncpu > 1:
        pool = Pool(ncpu)

    print("    Computing TMscores...")

    if method=="maxsub":
        print("compute dmat using maxsub...")

        scores = []
        for n in range(1,numrep+1):
            repn = "rep%d"%n
            for i in range(nf):
                pdbi = "/tmp/mdfolder/%s_md_%s_%d.pdb"%(pref,repn,i)
                print("  %-45s <--> %s"%(pdbi,fpdblist))
                if ncpu>1:
                    pool.apply_async(maxsub,
                                     (pdbi,fpdblist,numrep,repn,nf,i,metric,),
                                     callback=scores.append)
                else:
                    scores.append(maxsub(pdbi,fpdblist,numrep,repn,nf,i,metric))
        if ncpu>1:
            pool.close()
            pool.join()

        #print(tmscores)
        scores = sorted(scores,key=lambda x: (x[2],x[3]))
        #print(len(scores),numrep,nf)
        tmscores = np.matrix([scores[i][0] for i in range(numrep*nf)])
        rmsds = np.matrix([scores[i][1] for i in range(numrep*nf)])
        
        #print(dmat)
        if metric == "TM":
            dmat = 1.0 - tmscores
        else:
            dmat = rmsds

    np.savetxt(ftm,tmscores)
    np.savetxt(frmsd,rmsds)
    np.savetxt(fmat,dmat)
    return dmat

def dens_cut(dij,dcut):
    den = dij < dcut
    return den

def dens_exp(dij,dcut):
    dij2 = dij*dij
    dc2  = dcut*dcut
    den  = np.exp(-3*dij2/dc2)
    return den

def get_gamma(nf,dmat,density):
    delta = [0 for i in range(nf)]
    ordered_density = [[i,density[i]] for i in range(nf)]
    ordered_density = sorted(ordered_density,key=lambda x:x[1])
    #print(ordered_density)
    for i in reversed(range(nf-1)):
        dmin = 1e10
        idx1,den1 = ordered_density[i]
        delta[idx1] = dmin
        for j in range(i+1,nf):
            idx2,den2 = ordered_density[j]
            if den2>=den1:
            #print(idx1,idx2,dmat[idx1,idx2])
                if dmat[idx1,idx2] < dmin:
                    dmin = dmat[idx1,idx2]
                    delta[idx1] = dmin

    imax,max_dens = ordered_density[-1]
    print(sorted(density,reverse=True)[0:10])
    dmax = 0
    for j in range(nf):
        if dmat[imax,j]>dmax:
            dmax = dmat[imax,j]
    delta[imax] = dmax

    #print(density)
    gamma = [density[i]*delta[i] for i in range(nf)]
    #print(gamma)

    return gamma


def DPclustering(dmat,npeak=10,dcut=0.5):
    nf, nf  = np.shape(dmat)
    #dcut = 0.05
    #density = [np.sum(dens_exp(dmat[i,:],dcut)) for i in range(nf)]
    dcut = 0.02
    density = [np.sum(dens_cut(dmat[i,:],dcut)) for i in range(nf)]
    while np.mean(density)<0.03*nf:
        dcut += 0.02
        #density = [np.sum(dens_exp(dmat[i,:],dcut)) for i in range(nf)]
        density = [np.sum(dens_cut(dmat[i,:],dcut)) for i in range(nf)]
    print(dcut)
    #print("    Computing gamma...")
    gamma   = get_gamma(nf,dmat,density)
    ordered_gamma = [[i,gamma[i]] for i in range(nf)]
    ordered_gamma = sorted(ordered_gamma,key=lambda x:x[1],reverse=True)
    peaks = [ordered_gamma[s][0] for s in range(npeak)]
    print([od[1] for od in ordered_gamma[0:npeak]])
    print(peaks)

    return peaks
    

def get_peaks(pref,numrep,t1,t2,dt,redo=False,npeak=5,pcut=0.5,ncpu=1,metric="TM",mdfolder="./"):
    
    tpr  = "./%s_em0.pdb"%(pref)
    fpdblist = "./%s_pdblist.txt"%(pref)
    #print("Computing distance matrix...")
    start_time = time.time()
    dmat = get_dmat(pref,numrep,fpdblist,tpr,t1,t2,dt,
                    mdfolder,redo,metric=metric,ncpu=ncpu)
    #print("Finished distance matrix in %.2f seconds"%(time.time()-start_time))

    #print("Computing peaks...")
    start_time = time.time()
    #dcut    = np.quantile(dmat,pcut)
    #dmin = np.min(dmat[np.nonzero(dmat)])
    #dmax = np.max(dmat[np.nonzero(dmat)])
    #if metric == "TM":
    #    #dcut = dmin + 0.1*(dmax-dmin)
    #    dcut = np.quantile(dmat,pcut)
    #elif RMSD == "RMSD":
    #    dcut = dcut + 0.2
    dcut = 0.05
    peaks = DPclustering(dmat,npeak=npeak,dcut=dcut)
    #print(peaks)

    nf = int((t2-t1)/dt+1)
    for i in range(npeak):
        idx  = peaks[i]
        rep_idx = idx//nf+1
        trj_idx = idx%nf
        repn = "rep%d"%rep_idx
        pdb1 = "/tmp/mdfolder/%s_md_%s_%d.pdb"%(pref,repn,trj_idx)
        pdb2 = "./peaks/%s_peak%d.pdb"%(pref,i)
        shutil.copy(pdb1,pdb2)

    rmsd_fl   = []
    rmsd_core = []
    gdt_score = []
    tm_score  = []
    for i in range(npeak):
        pdbi = "./peaks/%s_peak%d.pdb"%(pref,i)
        for j in range(i+1,npeak):
            pdbj = "./peaks/%s_peak%d.pdb"%(pref,j)
            #cmd = "TMalign %s %s -byresi 1"%(pdbi,pdbj)
            #result = subprocess.check_output(cmd,
            #            shell=True).decode(sys.stdout.encoding)
            #info = result.split()
            #rmsd_core.append(float(info[5][:-1]))
            #rmsd_fl.append(float(info[10][:-1]))
            #tm_score.append(float(info[11][3:-1]))

            gdt,tm,rmsd = tmalign(pdbi,pdbj,byresi=False)
            tm_score.append(tm)
            gdt_score.append(gdt)
            rmsd_core.append(rmsd)

            gdt,tm,rmsd = tmalign(pdbi,pdbj,byresi=True)
            rmsd_fl.append(rmsd)
            #print(pdbi,pdbj,gdt,tm,rmsd)

    gdt_score = np.mean(gdt_score)
    rmsd_core = np.mean(rmsd_core)
    rmsd_fl   = np.mean(rmsd_fl)
    tm_score  = np.mean(tm_score)
    res = pref,rmsd_fl,rmsd_core,gdt_score,tm_score
    print("%s %.2f %.2f %.2f %.2f"%tuple(res))

    #print("Finished peaks in %.2f seconds"%(time.time()-start_time))
    #print(peaks)
    return rmsd_fl,rmsd_core,gdt_score,tm_score

def main(flist,numrep,t1,t2,dt,metric="TM",mdfolder="./",pcut=0.2,ncpu=1,npeak=5,redo=False):
    data = {
            "FBID" : [],
            "Gene": [],
            "RMSD_FL" : [],
            "RMSD_CORE" :[],
            "GDT": [],
            "TM-score":[],
            }

    #checklist = []
    #lines = open("checklist.txt","r")
    #for line in lines:
    #    checklist.append(line.strip())
    #print(checklist)
    genenames = get_genenames()
    
    lines = open(flist,"r")
    for line in lines:
        pref = line.split()[0]
        #if pref in checklist:
        #    redo = True
        #else:
        #    redo = False
        #if pref == "FBgn0085330":
        #    redo = False
        #else:
        redo = True
        res  = get_peaks(pref,numrep,t1,t2,dt,redo=redo,mdfolder=mdfolder,
                    pcut=pcut,ncpu=ncpu,metric=metric,npeak=npeak)
        rmsd_fl,rmsd_core,gdt_score,tm_score = res
        data["FBID"].append(pref)
        data["Gene"].append(genenames[pref[0:11]])
        data["RMSD_FL"].append(rmsd_fl)
        data["RMSD_CORE"].append(rmsd_core)
        data["GDT"].append(gdt_score)
        data["TM-score"].append(tm_score)
        #break

    data = pd.DataFrame(data)
    data.to_csv("mdsimulations_cluster_3reps_combined.csv",index=False)


if __name__ == "__main__":
    dpeaks   = "./peaks/"
    tmpfolder = "/tmp/mdfolder/" 
    if not os.path.isdir(dpeaks):
        os.mkdir(dpeaks)
    if not os.path.isdir(tmpfolder):
        os.mkdir(tmpfolder)
    flist  = "../asr_20221207.txt"
    #flist  = "finished.txt"
    metric = "TM"
    numrep = 3
    mdfolder= "./trajs_bak/"
    redo = True
    ncpu = 36
    t1   = 100.0
    t2   = 200.0
    #t2   = 60.0
    #dt   = 1
    dt   = 0.1
    pcut = 0.05
    npeak = 5
    start_time = time.time()
    main(flist,numrep,t1,t2,dt,metric="TM",mdfolder=mdfolder,
                pcut=pcut,ncpu=ncpu,npeak=npeak,redo=redo)

    #pref = "FBgn0262896"
    #pref = "FBgn0014850"
    #pref = "FBgn0264746"
    #pref = "FBgn0264747"
    #pref = "FBgn0037042"
    #pref = "FBgn0263250"
    #pref = "FBgn0261581"
    #pref = "FBgn0262896_L2"
    #get_peaks(pref,numrep,t1,t2,dt,mdfolder=mdfolder,
    #            redo=redo,pcut=pcut,ncpu=ncpu,metric=metric)
    #print("Total time: %.2f seconds"%(time.time()-start_time))
