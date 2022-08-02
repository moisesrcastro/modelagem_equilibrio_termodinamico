import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from KIJ import Kij_otimizado
'''Equilibrio Líquido Vapor Sistema: Etano(1) - Propano(2)'''

# Constantes
R = 0.08205;    T = 333.15;

# Propriedades critica dos fluídos

'''Etano'''         '''Propano'''
Pc1 = 48.2;         Pc2 = 41.9;
Tc1 = 305.5;        Tc2 = 369.8;
w1 = 0.097;         w2 = 0.152;

#Kij = 0
'''Kij Otimizado'''
Kij = Kij_otimizado

# Dados Experimentais

X1 = np.array([0, 0.0711, 0.152, 0.23, 0.305, 0.378, 0.448, 0.515, 0.581, 0.613, 0.652, 0.6685])
Y1 = np.array([0, 0.14, 0.267, 0.367, 0.448, 0.515, 0.573, 0.618, 0.655, 0.672, 0.67, 0.6685])
P = np.array([20.983, 23.814, 27.216, 30.618, 34.02, 37.422, 40.824, 44.226, 47.628, 49.329, 51.030, 51.166])
X2 = 1 - np.transpose(X1);      Y2 = 1 - np.transpose(Y1);      N = len(X1);    P_exp = P;  Y1_exp = Y1;

def Equilibrio(Y1, Y2, X1, X2, Kij, P):

    for i in range(1, 3):

        globals()['Tr' + str(i)] = T / eval('Tc' + str(i));
        globals()['m' + str(i)] = 0.48 + 1.574 * eval('w' + str(i)) - 0.176 * (eval('w' + str(i)) ** 2);
        globals()['aT' + str(i)] = (1 + eval('m' + str(i)) * (1 - (eval('Tr' + str(i)) ** 0.5))) ** 2;
        globals()['a' + str(i)] = 0.45724 * ((R * eval('Tc' + str(i))) ** 2) / eval('Pc' + str(i)) * eval('aT' + str(i));
        globals()['b' + str(i)] = 0.0778 * R * eval('Tc' + str(i)) / eval('Pc' + str(i));

            #Calculo do aij

    for i in range(1, 3):
        for j in range(1, 3):
            globals()['a' + str(i) + str(j)] = (1-Kij)*np.sqrt(((eval('a' + str(i))) * (eval('a' + str(j)))))


    bL = 0;        bV = 0;        aL = 0;        aV = 0;
    for i in range(1, 3):
        bL += eval('X' + str(i)) * eval('b' + str(i))
        bV += eval('Y' + str(i)) * eval('b' + str(i))

        for j in range(1, 3):
            aL += eval('X' + str(i)) * eval('X' + str(j)) * eval('a' + str(i) + str(j))
            aV += eval('Y' + str(i)) * eval('Y' + str(j)) * eval('a' + str(i) + str(j))


    AL = aL * P / (R ** 2 * T ** 2)
    AV = aV * P / (R ** 2 * T ** 2)
    BL = bL * P / (T * R)
    BV = bV * P / (T * R)


    Z1 = [1, -(1 - BL), AL - 2 * BL - 3 * (BL ** 2), -AL * BL + (BL ** 2) + (BL ** 3)]
    Z2 = [1, -(1 - BV), AV - 2 * BV - 3 * (BV ** 2), -AV * BV + (BV ** 2) + (BV ** 3)]

    Zl = np.roots(Z1)
    Zv = np.roots(Z2)
    ZL = 0
    ZV = 0

    if sum(np.iscomplex(Zl)) == 2:
        for i in range(3):
            if np.isreal(Zl[i]):
                ZL += (np.real(Zl[i]))

    if sum(np.iscomplex(Zv)) == 2:
        for i in range(3):
            if np.isreal(Zv[i]):
                ZV += (np.real(Zv[i]))

    if sum(np.isreal(Zl)) == 3:
        ZL += (np.real(min(Zl)))

    if sum(np.isreal(Zv)) == 3:
        ZV += np.real(max(Zv))


    phiL = np.zeros(2)
    phiV = np.zeros(2)
    K = np.zeros(2)

    LnL = np.log((ZL + (1 - np.sqrt(2)) * BL) / (ZL + (1 + np.sqrt(2)) * BL))
    LnV = np.log((ZV + (1 - np.sqrt(2)) * BV) / (ZV + (1 + np.sqrt(2)) * BV))

    for i in range(1, 3):
        phiL[i - 1] = np.exp(eval('b' + str(i)) / bL * (ZL - 1) - np.log(ZL - BL) + AL / (2 * np.sqrt(2) * BL) * (2 * (X1 * eval('a1' + str(i)) + X2 * eval('a2' + str(i))) / aL - eval('b' + str(i)) / bL) * LnL)
        phiV[i - 1] = np.exp(eval('b' + str(i)) / bV * (ZV - 1) - np.log(ZV - BV) + AV / (2 * np.sqrt(2) * BV) * (2 * (Y1 * eval('a1' + str(i)) + Y2 * eval('a2' + str(i))) / aV - eval('b' + str(i)) / bV) * LnV)
        K[i - 1] = phiL[i - 1] / phiV[i - 1]

    return phiL, phiV, K

def GetValues(Y_and_P, X1, Kij):

    Y1 = Y_and_P[0]
    P = Y_and_P[1]
    Y2 = 1 - Y1
    X2 = 1 - X1

    phiL, phiV, K = Equilibrio(Y1, Y2, X1, X2, Kij, P)

    A = Y1 * phiV[0] * P - X1 * phiL[0] * P
    B = Y2 * phiV[1] * P - X2 * phiL[1] * P

    return [A, B]

y1_predict = []
p_predict = []

for i in range(N):

    ans = fsolve(GetValues, [Y1[i], P[i]], args=(X1[i], Kij))
    y1_predict.append(ans[0])
    p_predict.append(ans[1])

DesvioP =[];        DesvioY=[];

for i in range(N):

    desvioP = abs(p_predict[i]-P_exp[i])*100/P_exp[i]; DesvioP.append(desvioP)
    if Y1_exp[i]==0: DesvioY.append(0);
    else:
        desvioY = abs(y1_predict[i]-Y1_exp[i])*100/Y1_exp[i];DesvioY.append(desvioY)

'''             Criando as Tabelas           '''

Table = PrettyTable();desv = PrettyTable();
Table.add_column('Pressão',list(p_predict));Table.add_column('Y1',list(y1_predict));print(Table);
desv.add_column('Desvio Padrão P',[sum(DesvioP)/N]);desv.add_column('Desvio Padrão Y1',[sum(DesvioY)/N]);print(desv);

plt.xlabel('x1 , y1');plt.ylabel('P (atm)');plt.title('Dados Experimentais X Calculados');
plt.plot(X1,p_predict,'k--',color='lime',label='Curva Bolha Calculada');plt.legend();
plt.plot(y1_predict,p_predict,'r--',color='cyan',label='Curva Orvalho Calculada');plt.legend();
plt.plot(X1,P_exp,'go',color='green',label='Curva Bolha Experimental');plt.legend();
plt.plot(Y1_exp,P_exp,'bo',color='blue',label='Curva Orvalho Experimental');plt.legend();
plt.show()

