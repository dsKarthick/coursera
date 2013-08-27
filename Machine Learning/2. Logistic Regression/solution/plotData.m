function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure

num_cols = size(X,2);
idx = y==1;
plot(X(idx,1),X(idx,2),'k+','LineWidth',2,'MarkerSize',10);
hold on; 
idx = y==0;
plot(X(idx,1),X(idx,2),'yo','LineWidth',2,'MarkerSize',7,'MarkerFaceColor','y','MarkerEdgeColor','k');
% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%









% =========================================================================



hold off;

end
