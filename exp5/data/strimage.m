function strimage(n)
  fidin = fopen('train-01-images.svm'); % 打开test2.txt文件 
  i = 1
  apres = []

while ~feof(fidin)
  tline = fgetl(fidin); % 从文件读�? 
  apres{i} = tline;
  i = i+1;
end

  a = char(apres(n))
  
  lena = size(a);
  lena = lena(2);
  xy = sscanf(a(4:lena), '%d:%d');

  lenxy = size(xy);
  lenxy = lenxy(1);
  
  
  grid = [];
  grid(784) = 0;
  for i=2:2:lenxy  %% 隔一个数
      if(xy(i)<=0)
          break
      end
    grid(xy(i-1)) = 255;
  end
  grid1 = reshape(grid,28,28);
  grid1 = fliplr(diag(ones(28,1)))*grid1;
  grid1 = rot90(grid1,3)
  image(grid1)
  hold on;
end
