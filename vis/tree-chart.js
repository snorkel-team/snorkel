// See http://bl.ocks.org/d3noob/8375092
// Three vars need to be provided via python string formatting:
var chartId = "%s";
var root = %s;
var highlightIdxs = %s;

// Highlight words / nodes
var COLORS = ["#ff5c33", "#ffcc00", "#33cc33", "#3399ff"];
function highlightWords() {
  for (var i=0; i < highlightIdxs.length; i++) {
    var c = COLORS[i];
    var idxs = highlightIdxs[i];
    for (var j=0; j < idxs.length; j++) {
      d3.selectAll(".word-"+chartId+"-"+idxs[j]).style("stroke", c).style("background", c);
    }
  }
}

// Constants
var margin = {top: 20, right: 20, bottom: 20, left: 20},
width = 800 - margin.left - margin.right,
height = 350 - margin.top - margin.bottom,
R = 5;

// Create the d3 tree object
var tree = d3.layout.tree()
  .size([width, height]);

// Create the svg canvas
var svg = d3.select("#tree-chart-" + chartId)
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
    //.on("click", function() {
    //  d3.select(this).classed("highlight", !d3.select(this).classed("highlight")); })
    .attr("r", R)
    .attr("class", function(d) { return "word-"+chartId+"-"+d.attrib.word_idx; });
     
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
highlightWords();
