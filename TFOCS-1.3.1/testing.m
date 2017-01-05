function testing
cmat = xlsread('clustersprev.csv');
for i=1:6
    display(sum(cmat==i));
    display('\n');
end;
end