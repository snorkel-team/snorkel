require.undef('viewer');

// NOTE: all elements should be selected using this.$el.find to avoid collisions with other Viewers

define('viewer', ["@jupyter-widgets/base"], function(widgets) {
    var ViewerView = widgets.DOMWidgetView.extend({
        render: function() {
            this.cids   = this.model.get('cids');
            this.nPages = this.cids.length;
            this.pid  = 0;
            this.cxid = 0;
            this.cid  = 0;

            // Insert the html payload
            this.$el.append(this.model.get('html'));

            // Initialize all labels from previous sessions
            this.labels = this.deserializeDict(this.model.get('_labels_serialized'));
            for (var i=0; i < this.nPages; i++) {
                this.pid = i;
                for (var j=0; j < this.cids[i].length; j++) {
                    this.cxid = j;
                    for (var k=0; k < this.cids[i][j].length; k++) {
                        this.cid = k;
                        if (this.cids[i][j][k] in this.labels) {
                            this.markCurrentCandidate(false);
                        }
                    }
                }
            }
            this.pid  = 0;
            this.cxid = 0;
            this.cid  = 0;

            // Enable button functionality for navigation
            var that = this;
            this.$el.find("#next-cand").click(function() {
                that.switchCandidate(1);
            });
            this.$el.find("#prev-cand").click(function() {
                that.switchCandidate(-1);
            });
            this.$el.find("#next-context").click(function() {
                that.switchContext(1);
            });
            this.$el.find("#prev-context").click(function() {
                that.switchContext(-1);
            });
            this.$el.find("#next-page").click(function() {
                that.switchPage(1);
            });
            this.$el.find("#prev-page").click(function() {
                that.switchPage(-1);
            });
            this.$el.find("#label-true").click(function() {
                that.labelCandidate(true, true);
            });
            this.$el.find("#label-false").click(function() {
                that.labelCandidate(false, true);
            });

            // Arrow key functionality
            this.$el.keydown(function(e) {
                switch(e.which) {
                    case 74: // j
                    that.switchCandidate(-1);
                    break;

                    case 73: // i
                    that.switchPage(-1);
                    break;

                    case 76: // l
                    that.switchCandidate(1);
                    break;

                    case 75: // k
                    that.switchPage(1);
                    break;

                    case 84: // t
                    that.labelCandidate(true, true);
                    break;

                    case 70: // f
                    that.labelCandidate(false, true);
                    break;
                }
            });

            // Show the first page and highlight the first candidate
            this.$el.find("#viewer-page-0").show();
            this.switchCandidate(0);
        },

        // Get candidate selector for currently selected candidate, escaping id properly
        getCandidate: function() {
            return this.$el.find("."+this.cids[this.pid][this.cxid][this.cid]);
        },  

        // Color the candidate correctly according to registered label, as well as set highlighting
        markCurrentCandidate: function(highlight) {
            var cid  = this.cids[this.pid][this.cxid][this.cid];
            var tags = this.$el.find("."+cid);

            // Clear color classes
            tags.removeClass("candidate-h");
            tags.removeClass("true-candidate");
            tags.removeClass("true-candidate-h");
            tags.removeClass("false-candidate");
            tags.removeClass("false-candidate-h");
            tags.removeClass("highlighted");

            if (highlight) {
                if (cid in this.labels) {
                    tags.addClass(String(this.labels[cid]) + "-candidate-h");
                } else {
                    tags.addClass("candidate-h");
                }
            
            // If un-highlighting, leave with first non-null coloring
            } else {
                var that = this;
                tags.each(function() {
                    var cids = $(this).attr('class').split(/\s+/).map(function(item) {
                        return parseInt(item);
                    });
                    cids.sort();
                    for (var i in cids) {
                        if (cids[i] in that.labels) {
                            var label = that.labels[cids[i]];
                            $(this).addClass(String(label) + "-candidate");
                            $(this).removeClass(String(!label) + "-candidate");
                            break;
                        }
                    }
                });
            }

            // Extra highlighting css
            if (highlight) {
                tags.addClass("highlighted");
            }

            // Classes for showing direction of relation
            if (highlight) {
                this.$el.find("."+cid+"-0").addClass("left-candidate");
                this.$el.find("."+cid+"-1").addClass("right-candidate");
            } else {
                this.$el.find("."+cid+"-0").removeClass("left-candidate");
                this.$el.find("."+cid+"-1").removeClass("right-candidate");
            }
        },

        // Cycle through candidates and highlight, by increment inc
        switchCandidate: function(inc) {
            var N = this.cids[this.pid].length
            var M = this.cids[this.pid][this.cxid].length;
            if (N == 0 || M == 0) { return false; }

            // Clear highlighting from previous candidate
            if (inc != 0) {
                this.markCurrentCandidate(false);

                // Increment the cid counter

                // Move to next context
                if (this.cid + inc >= M) {
                    while (this.cid + inc >= M) {
                        
                        // At last context on page, halt
                        if (this.cxid == N - 1) {
                            this.cid = M - 1;
                            inc = 0;
                            break;
                        
                        // Increment to next context
                        } else {
                            inc -= M - this.cid;
                            this.cxid += 1;
                            M = this.cids[this.pid][this.cxid].length;
                            this.cid = 0;
                        }
                    }

                // Move to previous context
                } else if (this.cid + inc < 0) {
                    while (this.cid + inc < 0) {
                        
                        // At first context on page, halt
                        if (this.cxid == 0) {
                            this.cid = 0;
                            inc = 0;
                            break;
                        
                        // Increment to previous context
                        } else {
                            inc += this.cid + 1;
                            this.cxid -= 1;
                            M = this.cids[this.pid][this.cxid].length;
                            this.cid = M - 1;
                        }
                    }
                }

                // Move within current context
                this.cid += inc;
            }
            this.markCurrentCandidate(true);

            // Push this new cid to the model
            this.model.set('_selected_cid', this.cids[this.pid][this.cxid][this.cid]);
            this.touch();
        },

        // Switch through contexts
        switchContext: function(inc) {
            this.markCurrentCandidate(false);

            // Iterate context on this page
            var M = this.cids[this.pid].length;
            if (this.cxid + inc < 0) {
                this.cxid = 0;
            } else if (this.cxid + inc >= M) {
                this.cxid = M - 1;
            } else {
                this.cxid += inc;
            }

            // Reset cid and set to first candidate
            this.cid = 0;
            this.switchCandidate(0);
        },

        // Switch through pages
        switchPage: function(inc) {
            this.markCurrentCandidate(false);
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
            this.cxid = 0;
            this.switchCandidate(0);
        },

        // Label currently-selected candidate
        labelCandidate: function(label, highlighted) {
            var c    = this.getCandidate();
            var cid  = this.cids[this.pid][this.cxid][this.cid];
            var cl   = String(label) + "-candidate";
            var clh  = String(label) + "-candidate-h";
            var cln  = String(!label) + "-candidate";
            var clnh = String(!label) + "-candidate-h";

            // Toggle label highlighting
            if (c.hasClass(cl) || c.hasClass(clh)) {
                c.removeClass(cl);
                c.removeClass(clh);
                if (highlighted) {
                    c.addClass("candidate-h");
                }
                this.labels[cid] = null;
                this.send({event: 'delete_label', cid: cid});
            } else {
                c.removeClass(cln);
                c.removeClass(clnh);
                if (highlighted) {
                    c.addClass(clh);
                } else {
                    c.addClass(cl);
                }
                this.labels[cid] = label;
                this.send({event: 'set_label', cid: cid, value: label});
            }

            // Set the label and pass back to the model
            this.model.set('_labels_serialized', this.serializeDict(this.labels));
            this.touch();
        },

        // Serialization of hash maps, because traitlets Dict doesn't seem to work...
        serializeDict: function(d) {
            var s = [];
            for (var key in d) {
                s.push(key+"~~"+d[key]);
            }
            return s.join();
        },

        // Deserialization of hash maps
        deserializeDict: function(s) {
            var d = {};
            var entries = s.split(/,/);
            var kv;
            for (var i in entries) {
                kv = entries[i].split(/~~/);
                if (kv[1] == "true") {
                    d[kv[0]] = true;
                } else if (kv[1] == "false") {
                    d[kv[0]] = false;
                }
            }
            return d;
        },
    });

    return {
        ViewerView: ViewerView
    };
});
