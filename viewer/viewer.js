require.undef('viewer');

// NOTE: all elements should be selected using this.$el.find to avoid collisions with other Viewers

define('viewer', ["jupyter-js-widgets"], function(widgets) {
    var ViewerView = widgets.DOMWidgetView.extend({
        render: function() {
            this.nPages = this.model.get('n_pages');
            this.pid = 0;
            this.cid = 0;

            // Insert the html payload
            this.$el.append(this.model.get('html'));

            // Enable button functionality for navigation
            var that = this;
            this.$el.find("#next-cand").click(function() {
                that.switchCandidate(1);
            });
            this.$el.find("#prev-cand").click(function() {
                that.switchCandidate(-1);
            });
            this.$el.find("#next-page").click(function() {
                that.switchPage(1);
            });
            this.$el.find("#prev-page").click(function() {
                that.switchPage(-1);
            });
            this.$el.find("#label-true").click(function() {
                that.labelCandidate(true);
            });
            this.$el.find("#label-false").click(function() {
                that.labelCandidate(false);
            });

            // Arrow key functionality
            $(document).keydown(function(e) {

                // Check that the Jupyter notebook cell the viewer is in is selected
                if (this.$el.find("#viewer").parents(".cell").hasClass("selected")) {
                    switch(e.which) {
                        case 74: // j
                        this.switchCandidate(-1);
                        break;

                        case 73: // i
                        this.switchPage(-1);
                        break;

                        case 76: // l
                        this.switchCandidate(1);
                        break;

                        case 75: // k
                        this.switchPage(1);
                        break;
                    }
                }
            });

            // Show the first page and highlight the first candidate
            this.$el.find("#viewer-page-0").show();
            this.switchCandidate(0);

            // Enable tooltips
            // TODO: Fix this?
            //$(function () {
            //      this.$el.find('[data-toggle="tooltip"]').tooltip()
            //})
        },

        // Highlight spans
        setRGBABackgroundOpacity: function(el, opacity) {
            var rgx = /rgba\((\d+),\s*(\d+),\s*(\d+),\s*(\d\.\d+)\)/;
            var m   = rgx.exec(el.css("background-color"));

            // Handle rgb
            if (m == null) {
                rgx = /rgb\((\d+),\s*(\d+),\s*(\d+)\)/;
                m   = rgx.exec(el.css("background-color"));
            }
            el.css("background-color", "rgba("+m[1]+","+m[2]+","+m[3]+","+opacity+")");
        },

        // Cycle through candidates and highlight, by increment inc
        switchCandidate: function(inc) {
            var nC = parseInt(this.$el.find("#viewer-page-"+this.pid).attr("data-nc"));
            if (nC == 0) { return false; }

            // Clear highlighting from previous candidate
            var cOld = this.$el.find("span.c-"+this.pid+"-"+this.cid);
            this.setRGBABackgroundOpacity(cOld, "0.3");
            cOld.removeClass("highlighted-candidate");

            // Increment
            if (this.cid + inc < 0) {
                this.cid = nC + (this.cid + inc);
            } else if (this.cid + inc > nC - 1) {
                this.cid = (this.cid + inc) - nC;
            } else {
                this.cid += inc;
            }

            // Highlight new candidate
            var cNew = this.$el.find("span.c-"+this.pid+"-"+this.cid);
            this.setRGBABackgroundOpacity(cNew, 1.0);
            cNew.addClass("highlighted-candidate");

            // Fill in caption
            // TODO: Redo this with widget data?
            //this.$el.find("#candidate-caption").html(this.$el.find("#cdata-"+this.pid+"-"+this.cid).attr("caption"));
        },

        // Switch through pages
        switchPage: function(inc) {
            this.$el.find(".viewer-page").hide();
            if (this.pid + inc < 0) {
                this.pid = 0;
            } else if (this.pid + inc > this.nPages - 1) {
                this.pid = this.nPages - 1;
            } else {
                this.pid += inc;
            }
            this.$el.find("#viewer-page-"+this.pid).show();

            // Show pagination
            this.$el.find("#page").html(this.pid);

            // Reset cid and set to first candidate
            this.cid = 0;
            this.switchCandidate(0);
        },

        // Label candidates
        labelCandidate: function(label) {
            var cl  = String(label) + "-candidate";
            var cln = String(!label) + "-candidate";
            var c   = this.$el.find("span.c-"+this.pid+"-"+this.cid);
            if (c.hasClass(cl)) {
                c.removeClass(cl);
            } else {
                c.removeClass(cln);
                c.addClass(cl);
            }
        },
    });

    return {
        ViewerView: ViewerView
    };
});
