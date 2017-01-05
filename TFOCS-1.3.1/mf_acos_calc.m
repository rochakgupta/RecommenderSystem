function [arr] = mf_acos_calc(index) 

Y = xlsread('matrixacos.csv');
cmat = xlsread('clustersprev.csv');
vals = csvread('finalacosval.csv');
users = find(cmat==index);
N = length(users);
Y = Y(users,users);
c = 1;
omega = [];

temp = isnan(Y);
for i=1:N
    for j=1:N
        if(temp(i,j)==0)
            omega = [omega, c];
        end;
    c = c+1;
    end;
end;

observations = Y(omega);    % the observed entries
mu           = .001;        % smoothing parameter

tic
Xk = solver_sNuclearBP( {N,N,omega}, observations, mu );
toc


Rt = xlsread('ratings.csv');
ex = csvread('check.csv');
ex = ex+1;
Rt = Rt(users,ex);
fprintf('\n');
arr = zeros(5,2);
[Xk, Z]=sort(Xk,2,'descend');

for k=1:5
    c=0;
    for i=1:N
        for j=1:242
            if(Rt(i,j)>0)
                cnt=sum(Rt(:,j)>0);
                cnt=cnt-1;
                %fprintf('Users for i %d and j %d : %d\n',i,j,cnt);
                cnt=floor(0.2*k*cnt);
                SimS=0;
                Rs=0;
                f=cnt;
                c=c+1;
                for u=1:N
                    if(f==0)
                        break;
                    end;
                    if(Z(i,u)~=i && Rt(Z(i,u),j)>0)
                        f=f-1;
                        Rs = Rs + Rt(u,j)*Xk(i,u);
                        SimS = SimS + Xk(i,u);
                    end;
                end;
                if(SimS==0)
                    c=c-1;
                    continue;
                end;
                rating = Rs/SimS;
                arr(k,1) = arr(k,1) + (Rt(i,j)-rating)*(Rt(i,j)-rating);
                arr(k,2) = arr(k,2) + abs(Rt(i,j)-rating);
            end;
        end;
    end
    if(c>0)
        arr(k,1)=sqrt(arr(k,1)/c);
        arr(k,2)=arr(k,2)/c;
        vals(k,index)=c;
    end;
end;
dlmwrite('finalacosval.csv',vals,',');
display('Res:\n');
display(arr)

end