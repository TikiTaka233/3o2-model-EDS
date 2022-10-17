% I modify the derivative of density function and fv.
function s=NewtonIterate_up(xv,zeta_CIR,sigma_ini,lambda,Delta,q_CIR,TOL,eps)     % xv denotes initial value. We can change zeta_CIR and Delta by other parameters.
fv=zeta_CIR*exp(-zeta_CIR*(exp(sigma_ini)*exp(-lambda*Delta)+exp(xv)))*(exp(xv)/(exp(sigma_ini)*exp(-lambda*Delta)))^(q_CIR/2)*exp(xv)*besseli(q_CIR,2*zeta_CIR*exp(-1/2*lambda*Delta)*sqrt(exp(sigma_ini+xv)),1)*exp(abs(real(2*zeta_CIR*exp(-1/2*lambda*Delta)*sqrt(exp(sigma_ini+xv)))))-TOL;

if fv<=0;
    while(fv<=0)
    xv=xv-0.05;
    fv=zeta_CIR*exp(-zeta_CIR*(exp(sigma_ini)*exp(-lambda*Delta)+exp(xv)))*(exp(xv)/(exp(sigma_ini)*exp(-lambda*Delta)))^(q_CIR/2)*exp(xv)*besseli(q_CIR,2*zeta_CIR*exp(-1/2*lambda*Delta)*sqrt(exp(sigma_ini+xv)),1)*exp(abs(real(2*zeta_CIR*exp(-1/2*lambda*Delta)*sqrt(exp(sigma_ini+xv)))))-TOL;
    end
    s=xv+0.05;
    return
else
    
uzeta=zeta_CIR*exp(sigma_ini-lambda*Delta);
fv_der=-((zeta_CIR*exp(xv)-q_CIR-1)*besseli(q_CIR,2*sqrt(zeta_CIR*exp(xv)*uzeta),1)*exp(abs(real(2*sqrt(zeta_CIR*exp(xv)*uzeta))))-1/2*2*sqrt(zeta_CIR*exp(xv)*uzeta)*besseli((q_CIR+1),2*sqrt(zeta_CIR*exp(xv)*uzeta),1)*exp(abs(real(2*sqrt(zeta_CIR*exp(xv)*uzeta)))))*zeta_CIR*exp(-uzeta-zeta_CIR*exp(xv)+xv)*(zeta_CIR*exp(xv)/uzeta)^(q_CIR/2);
xzero=-fv/fv_der;
while (norm(xzero)>=eps&&fv>0)
    xv=xv+xzero;
fv=zeta_CIR*exp(-zeta_CIR*(exp(sigma_ini)*exp(-lambda*Delta)+exp(xv)))*(exp(xv)/(exp(sigma_ini)*exp(-lambda*Delta)))^(q_CIR/2)*exp(xv)*besseli(q_CIR,2*zeta_CIR*exp(-1/2*lambda*Delta)*sqrt(exp(sigma_ini+xv)),1)*exp(abs(real(2*zeta_CIR*exp(-1/2*lambda*Delta)*sqrt(exp(sigma_ini+xv)))))-TOL;
% uzeta=zeta_CIR*exp(sigma_ini-lambda*Delta);     % uzeta is not changed.
fv_der=-((zeta_CIR*exp(xv)-q_CIR-1)*besseli(q_CIR,2*sqrt(zeta_CIR*exp(xv)*uzeta),1)*exp(abs(real(2*sqrt(zeta_CIR*exp(xv)*uzeta))))-1/2*2*sqrt(zeta_CIR*exp(xv)*uzeta)*besseli(q_CIR+1,2*sqrt(zeta_CIR*exp(xv)*uzeta),1)*exp(abs(real(2*sqrt(zeta_CIR*exp(xv)*uzeta)))))*zeta_CIR*exp(-uzeta-zeta_CIR*exp(xv)+xv)*(zeta_CIR*exp(xv)/uzeta)^(q_CIR/2);
xzero=-fv/fv_der;   % update xzero
end
s=xv;
return
end