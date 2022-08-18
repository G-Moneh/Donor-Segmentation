#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np # {
import pandas as pd # Needed for (laplacian) }
import scipy.linalg as sl # Needed for (eigenstuff)
import matplotlib.pyplot as plt # Needed for plotting
from random import randint # Needed for creating list of random colors FOR the edges of Map
import warnings # { Needed to ignore the warning about losing imaginary part FOR the EigenStuff results
warnings.simplefilter("ignore", np.ComplexWarning) # }
import copy # Needed to prevent copy baseRDL from messing up
from matplotlib import ticker # Needed to adjust y-ticks FOR Graph function


# In[ ]:


def Graph(sorted_EigenValues):
    fig = plt.figure(figsize = (12, 8))
    ax = plt.axes()
    plt.plot([j.real for j in sorted(sorted_EigenValues, reverse = True)])
    plt.xlabel('Position of Eigen Value in Ordered List', size = 14)
    plt.ylabel('Eigen Value', size = 14)
    plt.title('Eigen Values of Laplacian Matrix (descending order)', size = 18)
    plt.grid()
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())


# In[ ]:


def EigenStuff(Matrix):  # input: laplacian; output: Fiedler Value, Fiedler Vector, Sorted Eigen Values, Sorted Eigen Vectors
    df = pd.DataFrame()
    SLeM = sl.eig(Matrix)
    sort_ind = np.argsort(SLeM[0])
    Evectors = pd.DataFrame(SLeM[1])
    sort_Evectors = df.append(Evectors[[i for i in sort_ind]]).astype(float)
    return [np.round(sorted(SLeM[0])[1].real, 4),
            np.round(sort_Evectors.iloc[:, 1], 4),
            np.round([j.real for j in sorted(SLeM[0])], 4),
            np.round(sort_Evectors, 4)]


# In[11]:


def Laplacian(df, connections = 2, minDonations = 3, matchALL = True, LaplacianMap = True, LaplacianMapFig = False, printResults = True):
    EdgeCount = 0 # Counts the number of connections and thus the number of edges on the map
    donorID = {} # {START
    IDdonor = {}
    for i in range(df.shape[0]): # -- Creating dictionaries for implicit index and donor labels
        donorID[df.iloc[i, 0]] = i
        IDdonor[i] = df.iloc[i, 0] # }END
        
    dfc = df.iloc[:, 1:] # Removes donor label column (currently required to be first)
    l_DR1 = dfc.shape[0] # Amount of donors being analyzed
    l_RD = dfc.shape[1] # Amount of donations being analyzed
    MD_L = [] # List of donors with too much missing data, implicit index
    maxMissing = l_RD - minDonations # Maximum missing data
    
    for i in range(l_DR1): # {START
        if dfc.iloc[i].isna().sum() > maxMissing: # -- Making the list of donors with too much missing data
            MD_L += [i] # }End
    dfc.drop(MD_L, inplace = True) # }END   Removes unqualified rows, by implicit index
    
    
    l_DR2 = dfc.shape[0] # Getting new amount of donors after removal
    for i in range(l_DR2): # {START
        for j in range(l_RD):
            if pd.isnull(dfc.iat[i, j]): # Replacing the null values with unique str values
                    dfc.iat[i, j] = str(i + j*np.pi) # }END
    
    
    Donor_Deg = [0]*l_DR2 # Creating a list of zeroes, length equal to the amount of donors
    Matrix = np.zeros((l_DR2, l_DR2)) # Creating zero matrix of size n = amount of donors
    RecDonee = [] # PROBABLY DELETE###############################################################################################################
    
    for i in range(l_DR2): # {START
        RecDonee = RecDonee + [list(df.iloc[i])] # }END   NO -- MAYBE NOT DELETE ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    
    for i in range(l_DR2): # {START
        for j in range(l_DR2): # {{START
            if j <= i: # Checks if already checked (since i and j are the same)
                pass
            else:
                count = MatchCount(RecDonee[i], RecDonee[j]) # MatchCounts counts the number of pairs between two donors (a, a, a  ->  b, a, a  =  2 matches)
                if count >= connections: # connections is DEFAULT 2
                        Matrix[i][j] = -1 # Adjusts the 2nd donor's cell of 1st donor's row to -1, marking a connection
                        Matrix[j][i] = -1 # Symmetry requires the reverse of above
                        Donor_Deg[i] += 1 # Plus 1 to the degree of the row with connection, again symmetry
                        Donor_Deg[j] += 1 # }}END
         # }END
        
                
    for i in range(len(Donor_Deg)): # {START
        Matrix[i][i] = Donor_Deg[i] # }END   Adjusts remaining zeroes in matrix (diagonal) to degree of the donor (amount of connections)
        
    
    for i in range(Matrix.shape[0]): # {START
        for j in range(Matrix.shape[0]): # {{START
            if j <= i: 
                pass 
            else: 
                if Matrix[i][j] == -1: 
                    EdgeCount += 1 # }}END
        # }END
        
    EigStuffs = EigenStuff(Matrix)
    Fiedler = EigStuffs[1]
    Vector1 = EigStuffs[3].iloc[:, 2]
    Vector2 = EigStuffs[3].iloc[:, 3]
    
    FourGroups = FourGroupings(Fiedler, Vector1, IDdonor)
    
    if LaplacianMap:            
        DiagramFig = Diagram(Matrix, l_DR2, Fiedler, IDdonor, EdgeCount, LaplacianMapFig) # Calls for the Diagram
        

    Negatives, Positives, Zeroes = Groupings(l_DR2, Vector1, IDdonor)
    
    NegGroupsCommons = Connections(RecDonee, NegDonorGroups, donorID)
    PosGroupsCommons = Connections(RecDonee, PosDonorGroups, donorID)
            
    NegDonorGroups, NegFiedlerGroups, NegNoShare = Reconnect(Negatives[0], Negatives[1], Negatives[2], matchALL, connections, NegGroupsCommons)
    PosDonorGroups, PosFiedlerGroups, PosNoShare = Reconnect(Positives[0], Positives[1], Positives[2], matchALL, connections, PosGroupsCommons)
    
    if printResults:
        with pd.option_context('display.max_columns', 50, 'display.width', 90):
            print('Secondary Groupings: ')
            print('Negative Negative: \n')
            print(FourGroups[0], '\n')
            print('Positive Positive: \n')
            print(FourGroups[1], '\n')
            print('Negative Positive: \n')
            print(FourGroups[2], '\n')
            print('Positive Negative: \n')
            print(FourGroups[3], '\n')
            ##### --- #####
            print('_______________')
            ##### --- #####
            print('Total Donors: ', l_DR2)
            print('Total Connections: ', EdgeCount)
            ##### --- #####
            print('_______________')
            ##### --- #####
            print('\nPOSITIVE GROUP: \n', Positives[0], '\n\n\n')
            print('Groupings: \n', PosDonorGroups, '\n')
            print('Associated Fiedler Vector Values: \n', PosFiedlerGroups, '\n\n\n')
            if matchALL:
                if PosNoShare:
                    print('Unmatched Donors: \n', PosNoShare, '\n\n')
                else:
                    pass
            else:
                if PosNoShare:
                    print('Unmatched [Donor, Donations...]: \n')
                    for i in PosNoShare:
                        print('', RecDonee[donorID.get(i)], '\n')
                    print('\n')
                else:
                    pass
            for i in range(len(PosGroupsCommons)):
                print('\tGroup: ', i + 1, '\n', PosDonorGroups[i], '\n', '\tCommonalities: \n', PosGroupsCommons[i], '\n')
            ##### --- #####
            print('_______________')
            ##### --- #####
            print('\nNEGATIVE GROUP: \n', Negatives[0], '\n\n\n')
            print('Groupings: \n', NegDonorGroups, '\n')
            print('Associated Fiedler Vector Values: \n', NegFiedlerGroups, '\n\n\n')
            if matchALL:
                if NegNoShare:
                    print('Unmatched Donors: \n', NegNoShare, '\n\n')
                else:
                    pass
            else:
                if PosNoShare:
                    print('Unmatched [Donor, Donations...]: \n')
                    for i in NegNoShare:
                        print('', RecDonee[donorID.get(i)], '\n')
                    print('\n')
                else:
                    pass
            for i in range(len(NegGroupsCommons)):
                print('\tGroup: ', i + 1, '\n', NegDonorGroups[i], '\n', '\tCommonalities: \n', NegGroupsCommons[i], '\n')
            if Zeroes[0].empty:
                pass
            else:
                print('\nZERO GROUP: \n', Zeroes[0], '\n')
    else:
        pass
    
    NegDGGC = []
    PosDGGC = []
    for i in range(len(NegDonorGroups)):
        NegDGGC += [[NegDonorGroups[i], NegGroupsCommons[i]]]
    for i in range(len(PosDonorGroups)):
        PosDGGC += [[PosDonorGroups[i], PosGroupsCommons[i]]]
    
    if LaplacianMapFig:
        return Matrix, DiagramFig, (l_DR2, EdgeCount), (Negatives, NegDGGC, NegFiedlerGroups), (Positives, PosDGGC, PosFiedlerGroups)
    else:
        return Matrix, (l_DR2, EdgeCount), (Negatives, NegDGGC, NegFiedlerGroups), (Positives, PosDGGC, PosFiedlerGroups)


# In[16]:


def MatchCount(a, b):
    ac = a.copy()
    for i in range(len(b)): # {START
        try:
            ac.remove(b[i])
        except ValueError: # Pass if element of 'b' is not an element of 'ac'
            pass # }END
        
    return (len(b) - len(ac))


# In[ ]:


def Diagram(Matrix, l_DR, Fiedler, IDdonor, EdgeCount, LaplacianMapFig):
    fig = plt.figure(figsize = (11, 11)) # I believe this is 11x11 inches (may be wrong)
    ax = plt.axes()
    x = [] # List of x coordinates for the donors (around a circle)
    y = [] # ^^      y coordinates                               ^^
    colors = [] # List of random colors to color edges of map (number of colors = EdgeCount)

    for i in range(l_DR): # {START
        x += [(l_DR/4)*np.cos(i*2*np.pi/l_DR)] # Adding the x coordinates
        y += [(l_DR/4)*np.sin(i*2*np.pi/l_DR)] # Adding the y coordinates
        xy = [(l_DR/4)*np.cos(i*2*np.pi/l_DR), (l_DR/4)*np.sin(i*2*np.pi/l_DR)] # xy coordinates for annotating

        if np.cos(i*2*np.pi/l_DR) == 1: # x
            xy[0] = xy[0]*1.05
            plt.annotate('Donor ' + str(IDdonor.get(i)) + ':   ' + str(Fiedler[i]), xy, ha = 'left', va = 'center')
                                  # str(donorID.iloc[i])
                                  # str(i + 1)

        elif (0 < np.cos(i*2*np.pi/l_DR) < 1) and np.sin(i*2*np.pi/l_DR) > 0:  # 1st quadrant
            xy[0] = xy[0]*1.05
            xy[1] = xy[1]*1.05
            plt.annotate('Donor ' + str(IDdonor.get(i)) + ':   ' + str(Fiedler[i]), xy, ha = 'left', va = 'center')

        elif np.sin(i*2*np.pi/l_DR) == 1: # -- y
            xy[1] = xy[1]*1.05
            plt.annotate('Donor ' + str(IDdonor.get(i)) + ':   ' + str(Fiedler[i]), xy, ha = 'center', va = 'bottom')

        elif (-1 < np.cos(i*2*np.pi/l_DR) < 0) and np.sin(i*2*np.pi/l_DR) > 0:  # 2nd quadrant
            xy[0] = xy[0]*1.05
            xy[1] = xy[1]*1.05
            plt.annotate('Donor ' + str(IDdonor.get(i)) + ':   ' + str(Fiedler[i]), xy, ha = 'right', va = 'bottom')

        elif np.sin(i*2*np.pi/l_DR) == -1: # -- x
            xy[0] = xy[0]*1.05
            plt.annotate('Donor ' + str(IDdonor.get(i)) + ':   ' + str(Fiedler[i]), xy, ha = 'right', va = 'center')

        elif (-1 < np.cos(i*2*np.pi/l_DR) < 0) and np.sin(i*2*np.pi/l_DR) < 0:  # 3rd quadrant
            xy[0] = xy[0]*1.05
            xy[1] = xy[1]*1.05
            plt.annotate('Donor ' + str(IDdonor.get(i)) + ':   ' + str(Fiedler[i]), xy, ha = 'right', va = 'top')

        elif np.sin(i*2*np.pi/l_DR) == -1: # -- y
            xy[1] = xy[1]*1.05
            plt.annotate('Donor ' + str(IDdonor.get(i)) + ':   ' + str(Fiedler[i]), xy, ha = 'center', va = 'top')

        elif (0 < np.cos(i*2*np.pi/l_DR) < 1) and np.sin(i*2*np.pi/l_DR) < 0:  # 4th quadrant
            xy[0] = xy[0]*1.05
            xy[1] = xy[1]*1.05
            plt.annotate('Donor ' + str(IDdonor.get(i)) + ':   ' + str(Fiedler[i]), xy, ha = 'left', va = 'top') # }END


    for i in range(EdgeCount): # {START
        colors.append('#%06X' % randint(0, 0xFFFFFF)) # }End   Creating the list of colors previously mentioned

    for i in range(l_DR): # {START
        for j in range(l_DR): # {{START
            if j <= i:
                pass
            else:
                if Matrix[i][j] == -1:
                    ConnectPoints(x, y, i, j, colors)  # }}END   ConnectPoints graphs a line between two points
        # }END


    plt.scatter(x, y) # plots the lists of x and y coordinates (only dots, scatter)
    plt.title('MAP - Donor Connectivity')
    plt.axis('off') # Removes everything besides the actual graph
    if not LaplacianMapFig:
        plt.close(fig)
    return fig


# In[1]:


def ConnectPoints(x, y, p1, p2, colors):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    
    plt.plot([x1, x2], [y1, y2], color = colors[0])
    colors.remove(colors[0]) # Removal affects the original color list (copy was not made from input)


# In[2]:


def Groupings(l_DR, Fiedler, IDdonor):
    PosDF = pd.DataFrame()
    NegDF = pd.DataFrame()
    ZerDF = pd.DataFrame()
    
    NegDL = []
    PosDL = []
    ZerDL = []

    NegFL = []
    PosFL = []
    ZerFL = []
    
    for i in range(len(Fiedler)): # Creates lists by negative / positive (just in case: seperate group for zeroes)
        if Fiedler[i] > 0:
            PosDF.loc['values', str(IDdonor.get(i))] = Fiedler[i] # DF of all positive values
            PosDL.append(IDdonor.get(i)) # Donor List (anonymous identifier, label)
            PosFL.append(Fiedler[i]) # Associated Fiedler vector value (same shape as PosDL list)
        elif Fiedler[i] < 0:
            NegDF.loc['values', str(IDdonor.get(i))] = Fiedler[i]
            NegDL.append(IDdonor.get(i))                              # }
            NegFL.append(Fiedler[i])
        else:
            ZerDF.loc['values', str(IDdonor.get(i))] = Fiedler[i]
            ZerDL.append(IDdonor.get(i))                              # }
            ZerFL.append(Fiedler[i])
    if 'ZerDF' in locals():
        return (NegDF, NegDL, NegFL), (PosDF, PosDL, PosFL), (ZerDF, ZerDL, ZerFL)
    else:
        return (NegDF, NegDL, NegFL), (PosDF, PosDL, PosFL)


# In[3]:


def Reconnect(DF, DL, FL, matchALL, connections, GroupsCommons, DonorGroups):
    Already = []
    DonorGroups = []
    FiedlerGroups = []
    NoShareDG = []
    NoShareFG = []
    
    for i in range(len(DL)):                              # {{ i in range of length of Donor List
        if i not in Already:                              # Checks if i is marked (has already been checked / included)
            DonorGroups.append([DL[i]])                   # Creates new list (group) inside list of Groups
            FiedlerGroups.append([FL[i]])                 # }
            
        for j in range(len(DL)):                          # {{{ j in range of length of Donor List
            if j <= i or j in Already:                    # Checks if j is marked or has already been checked
                pass
            elif FL[i] == FL[j]:                          # Checks if Fiedler Vector values are equal
                for k in range(len(DonorGroups)):           ### Adding this fixed Neg issues but caused no reconnecting ###  
                    if FL[i] in FiedlerGroups[k]:            ##  It's because I was targeting DonorGroups mistakingly   ##
                        DonorGroups[k].append(DL[j])      # Puts j in group with i for Donor Groups 
                        FiedlerGroups[k].append(FL[j])    # }                      for Fiedler Groups
                        Already.append(j)                 # Marks j }}}
                                                          # }}
                            
    for i in range(len(GroupsCommons)):
        if len(GroupsCommons[i]) == 1:
            NoShareFG.
    
    for i in FiedlerGroups:
        if len(i) == 1:
            NoShareFG.append(i[0])
            FiedlerGroups.remove(i)
            
    for i in DonorGroups:
        if len(i) == 1:
            NoShareDG.append(i[0])                
            DonorGroups.remove(i)
        
                
    if matchALL:
        for i in range(len(FiedlerGroups)):                                   
            for j in range(len(NoShareFG)):
                cvlist = list(DF.iloc[0])
                cvlist.remove(NoShareFG[j])            
                if ClosestValue(cvlist, NoShareFG[j]) in FiedlerGroups[i]:
                    DonorGroups[i].append(NoShareDG[j])
                    FiedlerGroups[i].append(NoShareFG[j])
                    
    return (DonorGroups, FiedlerGroups, NoShareDG)


# In[5]:


def ClosestValue(input_list, input_value):
    arr = np.asarray(input_list)
    x = (np.abs(arr - input_value)).argmin()
    
    return arr[x]


# In[ ]:


def Connections(RecDonee, DonorGroups, donorID):
    GroupConnections = []
    for i in range(len(DonorGroups)):
        if np.size(DonorGroups[i]) > 1:
            DneA = RecDonee[donorID.get(DonorGroups[i][0])]
            DneB = RecDonee[donorID.get(DonorGroups[i][1])]
            CompL = []
        
            for j in DneB:
                try:
                    DneA.remove(j)
                    CompL.append(j)
                except ValueError:
                    pass

            for j in range(len(DonorGroups[i])):
                CompLremove = []
                if j < 2:
                    pass
                else:
                    DneJ = RecDonee[donorID.get(DonorGroups[i][j])] # ij inside list like [[1,18], *[16, 17, 2]*, [11, 19]]
                    for k in range(len(CompL)):
                        try:
                            DneJ.remove(CompL[k])
                        except ValueError:
                            CompLremove.append(CompL[k])
                for i in CompLremove:
                    CompL.remove(i)

            GroupConnections.append(CompL)
        else:
            pass
        
    return GroupConnections


# In[ ]:


def FourGroupings(Fiedler, Vector, IDdonor):
    NegNeg = []
    PosPos = []
    NegPos = []
    PosNeg = []
    
    for i in range(len(Fiedler)):
        if Fiedler[i] < 0:
            if Vector[i] < 0:
                NegNeg.append(IDdonor.get(i))
            else:
                NegPos.append(IDdonor.get(i))
        else:
            if Vector[i] > 0:
                PosPos.append(IDdonor.get(i))
            else:
                PosNeg.append(IDdonor.get(i))
    return (NegNeg, PosPos, NegPos, PosNeg)

