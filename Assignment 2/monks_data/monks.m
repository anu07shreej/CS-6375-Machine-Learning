for i = 1:3
    for s = {'train', 'test'}
        fName = sprintf('./monks-%d.%s.old', i, s{:});
        fID = fopen(fName, 'r');
        M = textscan(fID, '%d%d%d%d%d%d%d%s', 'delimiter', ',');
        fclose(fID);
        
        Z = cell2mat(M(1:7));
        fName = sprintf('./monks-%d.%s', i, s{:});
        csvwrite(fName, Z)
    end
end