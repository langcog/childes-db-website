# childes-db

The childes-db website is now generated using Jekyll, the static site generator typically used for Github Pages sites. The basic idea of a static site generator is that repetitive elements are specified once and included programatically; unlike typical dynamic web frameworks, these sites are rendered offline in a single batch process and only the static renders (output HTML pages) are served to clients. `_layouts` include templates with {{variables}}; `_includes` include repeated HTML or Markdown elements; `_site` includes the rendered site ; .html and .md elsewhere are converted into HTML and include a header to specify which template to use. 

Typically GH Pages runs Jekyll on all GH pages whenever a commit is pushed on master. To use Jekyll locally to preview the site (*strongly* recommended), you must install Jekyll following the instructions [here](https://jekyllrb.com/docs/installation/).

Install dependencies: `bundle install` 
Test build: `[bundle exec] jekyll build`
Serve on localhost: `[bundle exec] jekyll serve`

`jekyll serve` reflects live updates for any component files (templates, includes, or content).  

