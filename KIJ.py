import numpy as np
from scipy.optimize import fsolve

# Constantes

R = 0.08205;    T = 333.15;

# Propriedades critica dos fluídos

'''Etano'''         '''Propano'''
Pc1 = 48.2;         Pc2 = 41.9;
Tc1 = 305.5;        Tc2 = 369.8;
w1 = 0.097;         w2 = 0.152;

# Dados Experimentais

X1 = np.array([0, 0.0711, 0.152, 0.23, 0.305, 0.378, 0.448, 0.515, 0.581, 0.613, 0.652, 0.6685])
Y1 = np.array([0, 0.14, 0.267, 0.367, 0.448, 0.515, 0.573, 0.618, 0.655, 0.672, 0.67, 0.6685])
P = np.array([20.983, 23.814, 27.216, 30.618, 34.02, 37.422, 40.824, 44.226, 47.628, 49.329, 51.030, 51.166])
X2 = 1 - np.transpose(X1);      Y2 = 1 - np.transpose(Y1);      N = len(X1);    P_exp = P;  Y1_exp = Y1;

def Equilibrio(Kij):

    for i in range(1, 3):

        locals()['Tr' + str(i)] = T / eval('Tc' + str(i));
        locals()['m' + str(i)] = 0.48 + 1.574 * eval('w' + str(i)) - 0.176 * (eval('w' + str(i)) ** 2);
        locals()['aT' + str(i)] = (1 + eval('m' + str(i)) * (1 - (eval('Tr' + str(i)) ** 0.5))) ** 2;
        locals()['a' + str(i)] = 0.45724 * ((R * eval('Tc' + str(i))) ** 2) / eval('Pc' + str(i)) * eval('aT' + str(i));
        locals()['b' + str(i)] = 0.0778 * R * eval('Tc' + str(i)) / eval('Pc' + str(i));

        #Calculo do aij

    for i in range(1, 3):
        for j in range(1, 3):
            locals()['a' + str(i) + str(j)] = (1-Kij)*np.sqrt(((eval('a' + str(i))) * (eval('a' + str(j)))) )


    aL = [];    aV = [];
    bL = [];    bV = [];

    for k in range(N):
        bLi = 0;        bVi = 0;        aLi = 0;        aVi = 0;
        for i in range(1, 3):
            bLi += eval('X' + str(i))[k] * eval('b' + str(i))
            bVi += eval('Y' + str(i))[k] * eval('b' + str(i))

            for j in range(1, 3):
                aLi += eval('X' + str(i))[k] * eval('X' + str(j))[k] * eval('a' + str(i) + str(j))
                aVi += eval('Y' + str(i))[k] * eval('Y' + str(j))[k] * eval('a' + str(i) + str(j))

        aV.append(aVi);        aL.append(aLi);        bL.append(bLi);        bV.append(bVi);

        AL = [];        BL = [];
        AV = [];        BV = [];

    for k in range(N):
        AL.append(aL[k] * P[k] / (R ** 2 * T ** 2)); AV.append(aV[k] * P[k] / (R ** 2 * T ** 2));
        BL.append(bL[k] * P[k] / (T * R)); BV.append(bV[k] * P[k] / (T * R));

        ZL = [];        ZV = [];

    for i in range(N):
        Z1 = [1,-(1-BL[i]),AL[i]-2*BL[i]-3*(BL[i]**2),-AL[i]*BL[i]+(BL[i]**2)+(BL[i]**3)]
        Z2 = [1,-(1-BV[i]),AV[i]-2*BV[i]-3*(BV[i]**2),-AV[i]*BV[i]+(BV[i]**2)+(BV[i]**3)]

        Zl = np.roots(Z1)
        Zv = np.roots(Z2)
        if sum(np.iscomplex(Zl)) == 2:
            for i in range(3):
                if np.isreal(Zl[i]):
                    ZL.append(np.real(Zl[i]))
        if sum(np.iscomplex(Zv)) == 2:
            for i in range(3):
                if np.isreal(Zv[i]):
                    ZV.append(np.real(Zv[i]))

        if sum(np.isreal(Zl)) == 3:
            ZL.append(np.real(min(Zl)))
        if sum(np.isreal(Zv)) == 3:
            ZV.append(np.real(max(Zv)))


    phiL = np.zeros((N, 2))
    phiV = np.zeros((N, 2))
    K = np.zeros((N, 2))
    LnL = [];    LnV = [];

    for k in range(N):
        LnL.append(np.log((ZL[k] + (1 - np.sqrt(2)) * BL[k]) / (ZL[k] + (1 + np.sqrt(2)) * BL[k])))
        LnV.append(np.log((ZV[k] + (1 - np.sqrt(2)) * BV[k]) / (ZV[k] + (1 + np.sqrt(2)) * BV[k])))

        for i in range(1, 3):
            phiL[k, i - 1] = np.exp(eval('b' + str(i)) / bL[k] * (ZL[k] - 1) - np.log(ZL[k] - BL[k]) + AL[k] / (2 * np.sqrt(2) * BL[k]) * (2 * (X1[k] * eval('a1' + str(i)) + X2[k] * eval('a2' + str(i))) / aL[k] - eval('b' + str(i)) / bL[k]) * LnL[k])
            phiV[k, i - 1] = np.exp(eval('b' + str(i)) / bV[k] * (ZV[k] - 1) - np.log(ZV[k] - BV[k]) + AV[k] / (2 * np.sqrt(2) * BV[k]) * (2 * (Y1[k] * eval('a1' + str(i)) + Y2[k] * eval('a2' + str(i))) / aV[k] - eval('b' + str(i)) / bV[k]) * LnV[k])
            K[k, i - 1] = phiL[k, i - 1] / phiV[k, i - 1]

    F0 = sum(P * abs(phiV[:, 1] * Y2 - phiL[:, 1] * X2) + P * abs(phiV[:, 0] * Y1 - phiL[:, 0] * X1))
    return F0

answers = fsolve(Equilibrio, x0=0.0011)
Kij_otimizado = answers[0]
