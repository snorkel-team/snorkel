var vid = %s;
var cand = -1;

// Get the selector for a given candidate by number (w.r.t. to entire page group)
function cid(n) {
    return "span.c-" + n + "-" + vid;
}

// Get the total number of candidates on the page
function getNumberOfCandidates() {
    var nC = 0;
    $("li.context-li:visible").each(function() {
        nC += parseInt($(this).attr("data-nc"));
    });
    return nC;
}

// Cycle through candidates and highlight, by increment inc
function switchCandidate(inc) {
    var nC = getNumberOfCandidates();

    // Clear highlighting and highlight new candidate
    $("span.candidate").removeClass("highlight");
    if (cand + inc < 0) {
        cand = nC + (cand + inc);
    } else if (cand + inc > nC - 1) {
        cand = (cand + inc) - nC;
    } else {
        cand += inc;
    }
    var c = $(cid(cand));
    c.addClass("highlight");

    // Fill in caption
    $("#candidate-caption-"+vid).html($("#cdata-"+cand+"-"+vid).attr("caption"));
};

$("#next-cand-" + vid).click(function() {
    switchCandidate(1);
});

$("#prev-cand-" + vid).click(function() {
    switchCandidate(-1);
});

// Arrow key functionality
$(document).keydown(function(e) {

    // Check that the Jupyter notebook cell the viewer is in is selected
    if ($("#viewer-"+vid).parents(".cell").hasClass("selected")) {
        switch(e.which) {
            case 37: // Left
            switchCandidate(-1);
            break;

            case 38: // Up
            break;

            case 39: // Right
            switchCandidate(1);
            break;

            case 40: // Down
            break;
        }
    }
});
