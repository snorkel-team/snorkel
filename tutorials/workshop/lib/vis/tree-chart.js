// See http://bl.ocks.org/d3noob/8375092
// Two vars need to be provided via python string formatting: the tree, and the canvas id

// JSON tree
var root = %s;

// Constants
var margin = {top: 20, right: 20, bottom: 50, left: 20},
width = 800 - (margin.left - margin.right),
height = 300 - (margin.top - margin.bottom),
R = 5;

// Create the d3 tree object
var tree = d3.layout.tree()
  .size([width, height]);

// Create the svg canvas
var svg = d3.select("#tree-chart-%s")
  .append("svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
  .append("g")
  .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

function renderTree() {
  var nodes = tree.nodes(root),
  edges = tree.links(nodes);

  // Place the nodes
  var nodeGroups = svg.selectAll("g.node")
    .data(nodes)
    .enter().append("g")
    .attr("class", "node")
    .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
       
  // Append circles
  nodeGroups.append("circle")
    .on("click", function() {
      d3.select(this).classed("highlight", !d3.select(this).classed("highlight")); })
    .attr("r", R);
     
  // Append the actual word
  nodeGroups.append("text")
    .text(function(d) { return d.attrib.word; })
    .attr("text-anchor", function(d) { 
      return d.children && d.children.length > 0 ? "start" : "middle"; })
    .attr("dx", function(d) { 
      return d.children && d.children.length > 0 ? R + 3 : 0; })
    .attr("dy", function(d) { 
      return d.children && d.children.length > 0 ? 0 : 3*R + 3; });

  // Place the edges
  var edgePaths = svg.selectAll("path")
    .data(edges)
    .enter().append("path")
    .attr("class", "edge")
    .on("click", function() {
      d3.select(this).classed("highlight", !d3.select(this).classed("highlight")); })
    .attr("d", d3.svg.diagonal());
}

renderTree();
